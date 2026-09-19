from app import create_app

app = create_app()
client = app.test_client()

examples = [
    "Congratulations! You have won a free prize. Call now to claim your reward!",
    "Hi, the project meeting is moved to 3 pm tomorrow. Please bring the notes.",
]

for message in examples:
    response = client.post("/", data={"email_text": message})
    page = response.get_data(as_text=True)
    print("status=", response.status_code, "result_present=", "MODEL RESULT" in page)
    start = page.find('<h2>')
    end = page.find('</h2>', start)
    print("classification=", page[start + 4:end] if start >= 0 else "missing")
