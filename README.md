# UWU - Scrabble Board Image Analyzer

<b>How to run locally:</b>

    python ocr.py

It returns the tile position as a 15x15 array.

Blank tiles are predicted as well and are indicated as small letters.

<b>How to host:</b>

Run 'docker build' and deploy the image to AWS Lambda.

Send base64 encoded image to the Lambda API.

<b>Behind-the-scene processes:</b>

Board corner identification:
1. Apply Hough transform to identify vertical and horizontal straight lines of the image and record their intersection points.
2. For each quadrilaterals formed by the intersection points, project the inner intersection points formed by 15x15 squares of the board.
3. Prioritize quadrilaterals which encompass all the red regions (Triple Word Squares)
4. Count the number of points where the recorded intersection points align with the projected intersection points. Return best 3 quadrilaterals with highest count.

<img align="center" width="500" alt="Screenshot 2024-12-24 at 1 17 48 AM" src="https://github.com/user-attachments/assets/17162390-e09e-4f1a-9888-7f2707745e8a" />

Individual square identification:
1. Apply perspective transform to the quadrilateral to return a square board image.
2. Split the image by 15x15.
3. Use template matching to predict the most likely letter of the square.
4. After all tiles are predicted, identify the colour of the tiles.
5. Find the position of blanks where it has the 'same' colour of letter tiles but without a letter (i.e. blank)
6. Predict the letter represented by the blank by considering the adjacent letters (e.g. ? is P for A?PLE)

Returned array visualized:

<img align="center" width="500" alt="Screenshot 2024-12-24 at 1 55 44 AM" src="https://github.com/user-attachments/assets/7c1549f6-677a-4881-b9b2-0c1f7bbf2e55" />
