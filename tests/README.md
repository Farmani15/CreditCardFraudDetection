# Tests for Credit Card Fraud Detection

This directory contains test files for the credit card fraud detection system.

## Running Tests

To run all tests:

```bash
python -m unittest discover tests
```

To run a specific test file:

```bash
python -m unittest tests/test_model.py
```

## Test Files

- `test_model.py`: Tests for the model functionality, including data preprocessing, feature engineering, model training, and prediction.

## Adding New Tests

When adding new tests, follow these guidelines:

1. Create a new test file with the prefix `test_`.
2. Extend the `unittest.TestCase` class.
3. Write test methods with the prefix `test_`.
4. Use assertions to verify expected behavior.

Example:

```python
import unittest

class TestNewFeature(unittest.TestCase):
    def test_new_functionality(self):
        # Test code here
        self.assertEqual(expected_result, actual_result) 