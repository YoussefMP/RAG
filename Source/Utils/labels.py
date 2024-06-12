# Define the tag to ID mapping
TAG2ID = {"Section": 5, "subsection": 6, "Ref": 1,
          "Statue": 3, "Book": 2, "Line": 7,
          "Regulation": 8, "Article": 4,
          "O": 0, "Uncertain": 9,
          }

ID2TAG = {v: k for k, v in TAG2ID.items()}