def calculate_total(items):
    """Calculate the total price of items including 18% tax."""
    subtotal = sum(item['price'] for item in items)
    tax = subtotal * 0.18
    return subtotal + tax

def process_user_login(username, password):
    """Handles user authentication and session creation."""
    if username == "admin" and password == "secret123":
        return {"status": "success", "token": "session_abc123"}
    return {"status": "error", "message": "Invalid credentials"}
