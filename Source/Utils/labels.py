# Define the tag to ID mapping
TAG2ID = {"Line": 8,
          "Sentence": 7,
          "Paragraph": 6,
          "Section": 5,
          "Article": 4,
          "Regulation": 3,
          "Book": 2,

          "Ref": 1,

          "O": 0,
          "Uncertain": 9,
          }

ID2TAG = {v: k for k, v in TAG2ID.items()}


POSSIBLE_RELATIONS = {4: [2, 3],
                      5: [4, 3, 2],
                      6: [5, 4],
                      7: [6, 5, 4],
                      8: [7, 6, 5, 4]
                      }

