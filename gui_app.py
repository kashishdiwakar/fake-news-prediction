import tkinter as tk
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

SPAM_WORDS = [
    "lottery", "guaranteed", "free money", "miracle",
    "click here", "shocking", "breaking truth",
    "secret revealed", "you won't believe"
]

OPINION_WORDS = [
    "i think", "everyone knows", "believe me",
    "it seems", "probably", "might be"
]

FACT_WORDS = [
    "according to", "official", "report", "confirmed",
    "data shows", "statistics"
]

AUTHORITY_DOMAINS = {
    "isro": [
        "satellite", "launch", "rocket", "mission",
        "space", "orbit", "lunar", "mars",
        "payload", "gaganyaan"
    ],
    "rbi": [
        "interest rate", "repo rate", "bank",
        "currency", "inflation", "monetary policy"
    ],
    "who": [
        "health", "virus", "pandemic",
        "disease", "vaccination", "medical"
    ]
}

def authority_field_check(text):
    score = 0
    for authority, keywords in AUTHORITY_DOMAINS.items():
        if authority in text:
            if any(word in text for word in keywords):
                score += 25      
            else:
                score -= 40      
    return score

def analyze_news():
    text = input_box.get("1.0", tk.END).strip().lower()

    if not text:
        result_label.config(text="⚠️ Please enter news text", fg="red")
        return

    confidence_adjust = 0

    if any(word in text for word in SPAM_WORDS):
        confidence_adjust -= 25

    if any(word in text for word in OPINION_WORDS):
        confidence_adjust -= 10

    if any(word in text for word in FACT_WORDS):
        confidence_adjust += 15

    authority_score = authority_field_check(text)
    confidence_adjust += authority_score

    if authority_score < 0 and any(w in text for w in SPAM_WORDS):
        confidence_adjust -= 20

    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]      
    ml_confidence = max(model.predict_proba(vec)[0]) * 100

    final_confidence = ml_confidence + confidence_adjust
    final_confidence = max(0, min(100, final_confidence))

    if final_confidence < 40:
        result = "LIKELY FAKE"
        color = "red"

    elif 40 <= final_confidence < 60:
        result = "UNCERTAIN / VERIFY"
        color = "orange"

    elif 60 <= final_confidence < 80:
        if prediction == 0:
            result = "PROBABLY REAL"
            color = "green"
        else:
            result = "PROBABLY FAKE"
            color = "red"

    else:
        if prediction == 0 and authority_score > 0:
            result = "HIGHLY LIKELY REAL"
            color = "darkgreen"
        else:
            result = "SUSPICIOUS DESPITE AUTHORITY"
            color = "red"

    result_label.config(
        text=f"Result: {result}\nConfidence: {round(final_confidence, 2)}%",
        fg=color
    )
app = tk.Tk()
app.title("Advanced Fake News Detection System")
app.geometry("750x480")

tk.Label(
    app,
    text="Advanced Fake News Detector",
    font=("Arial", 18, "bold")
).pack(pady=10)

input_box = tk.Text(app, height=10, width=85)
input_box.pack(pady=10)

tk.Button(
    app,
    text="Analyze News",
    command=analyze_news,
    bg="blue",
    fg="white",
    font=("Arial", 12)
).pack(pady=10)

result_label = tk.Label(app, font=("Arial", 14), wraplength=700)
result_label.pack(pady=20)

app.mainloop()  