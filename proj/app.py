from flask import Flask, render_template, request
import pandas as pd
import math

app = Flask(__name__)
DATA_PATH = "bank_transactions.csv"

# ── FRAUD RULES ──────────────────────────────────────────
def apply_fraud_rules(df):
    df = df.copy()
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
    df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'], errors='coerce')
    df['SecondsSincePrev'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds().abs()

    df['R_HighLoginAttempts'] = df['LoginAttempts'] > 3
    df['R_ExceedsBalance']    = df['TransactionAmount'] > df['AccountBalance']
    df['R_FastTransaction']   = df['TransactionDuration'] < 30
    df['R_FrequentTx']        = df['SecondsSincePrev'] < 120

    df['IsFraud'] = (
        df['R_HighLoginAttempts'] | df['R_ExceedsBalance'] |
        df['R_FastTransaction']   | df['R_FrequentTx']
    ).astype(int)

    def build_reasons(row):
        reasons = []
        if row['R_HighLoginAttempts']:
            reasons.append(f"High login attempts ({int(row['LoginAttempts'])})")
        if row['R_ExceedsBalance']:
            reasons.append(f"Amount exceeds balance (${row['TransactionAmount']:.2f} > ${row['AccountBalance']:.2f})")
        if row['R_FastTransaction']:
            reasons.append(f"Fast transaction ({int(row['TransactionDuration'])} sec)")
        if row['R_FrequentTx']:
            secs = row['SecondsSincePrev']
            if pd.isna(secs):
                reasons.append("Frequent transaction (no previous date)")
            elif secs >= 60:
                reasons.append(f"Frequent transaction ({secs/60:.1f} min ago)")
            else:
                reasons.append(f"Frequent transaction ({secs:.0f} sec ago)")
        return " | ".join(reasons) if reasons else "—"

    df['FraudReasons'] = df.apply(build_reasons, axis=1)
    return df


# ── NAIVE BAYES MODEL ────────────────────────────────────
class NaiveBayesFraudModel:
    def fit(self, df):
        n = len(df)
        fraud = df[df['IsFraud'] == 1]
        safe  = df[df['IsFraud'] == 0]

        self.prior = {
            1: (len(fraud) + 1) / (n + 2),
            0: (len(safe)  + 1) / (n + 2),
        }

        self.channel_probs    = self._cat_probs(df, fraud, safe, 'Channel')
        self.occupation_probs = self._cat_probs(df, fraud, safe, 'CustomerOccupation')

        self.amount_stats = {
            1: {'mean': fraud['TransactionAmount'].mean(), 'std': fraud['TransactionAmount'].std()},
            0: {'mean': safe['TransactionAmount'].mean(),  'std': safe['TransactionAmount'].std()},
        }

    def _cat_probs(self, df, fraud, safe, col):
        probs = {}
        k = df[col].nunique()
        for val in df[col].dropna().unique():
            probs[(val, 1)] = (len(fraud[fraud[col] == val]) + 1) / (len(fraud) + k)
            probs[(val, 0)] = (len(safe[safe[col]   == val]) + 1) / (len(safe)  + k)
        return probs

    def _gauss(self, x, mean, std):
        std = std if (std and not pd.isna(std) and std != 0) else 0.01
        return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((x - mean) ** 2) / (2 * std ** 2))

    def predict(self, amount, channel, occupation):
        lk = {}
        for c in (1, 0):
            lk[c] = (
                self.prior[c]
                * self.channel_probs.get((channel, c), 0.5)
                * self.occupation_probs.get((occupation, c), 0.5)
                * self._gauss(amount, self.amount_stats[c]['mean'], self.amount_stats[c]['std'])
            )
        total = lk[0] + lk[1]
        return round((lk[1] / total) * 100, 2) if total else 0.0


# ── GLOBAL STATE ─────────────────────────────────────────
_df    = None
_error = None
model  = NaiveBayesFraudModel()

def get_data():
    global _df, _error
    if _df is not None or _error:
        return _df, _error
    try:
        df = pd.read_csv(DATA_PATH)
        df = apply_fraud_rules(df)
        model.fit(df)
        _df = df
    except FileNotFoundError:
        _error = f"Dataset '{DATA_PATH}' not found."
    except Exception as e:
        _error = str(e)
    return _df, _error


# ── ROUTE ────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    df, error = get_data()
    if error:
        return f"<h1>Error: {error}</h1>", 500

    total = len(df)
    fraud_count = int(df['IsFraud'].sum())
    fraud_rate  = round(fraud_count / total * 100, 2)

    stats = {
        'total_transactions': total,
        'fraud_transactions': fraud_count,
        'fraud_rate':  fraud_rate,
        'safe_rate':   round(100 - fraud_rate, 2),
        'avg_amount':  round(df['TransactionAmount'].mean(), 2),
        'max_amount':  round(df['TransactionAmount'].max(), 2),
        'channels_pct': {
            ch: round(cnt / total * 100, 2)
            for ch, cnt in df['Channel'].value_counts().items()
        },
        'rule_stats': {
            'high_login':  int(df['R_HighLoginAttempts'].sum()),
            'exceeds_bal': int(df['R_ExceedsBalance'].sum()),
            'fast_tx':     int(df['R_FastTransaction'].sum()),
            'frequent_tx': int(df['R_FrequentTx'].sum()),
        },
    }

    channels    = sorted(df['Channel'].dropna().unique())
    occupations = sorted(df['CustomerOccupation'].dropna().unique())

    view_type        = request.args.get('filter')
    min_amount       = request.args.get('min_amount')
    selected_channel = request.args.get('search_channel')

    transactions = []
    if view_type:
        fdf = df[df['IsFraud'] == 1].copy() if view_type == 'fraud' else df.copy()
        if min_amount:
            try: fdf = fdf[fdf['TransactionAmount'] >= float(min_amount)]
            except ValueError: pass
        if selected_channel and selected_channel != 'All':
            fdf = fdf[fdf['Channel'] == selected_channel]
        cols = ['TransactionID','AccountID','TransactionAmount','AccountBalance',
                'LoginAttempts','TransactionDuration','Channel','Location',
                'IsFraud','FraudReasons','SecondsSincePrev']
        transactions = fdf[cols].head(20).to_dict(orient='records')

    prediction = None
    if request.method == 'POST':
        try:
            prob = model.predict(
                float(request.form.get('amount', 0)),
                request.form.get('channel'),
                request.form.get('occupation'),
            )
            prediction = {
                'amount':      request.form.get('amount'),
                'channel':     request.form.get('channel'),
                'occupation':  request.form.get('occupation'),
                'probability': prob,
                'status': '🔴 HIGH RISK / FRAUDULENT' if prob >= 50 else '🟢 SAFE TRANSACTION',
            }
        except Exception as e:
            prediction = {'error': str(e)}

    return render_template('index.html',
        stats=stats, transactions=transactions,
        current_filter=view_type, channels=channels, occupations=occupations,
        prediction=prediction, min_amount=min_amount, search_channel=selected_channel,
    )


if __name__ == '__main__':
    app.run(debug=True)