class Invoice:
    def __init__(self, subtotal: int, tax: int | None = None):
        self.subtotal = subtotal
        self.tax = tax
