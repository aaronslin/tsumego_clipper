# tsumego_clipper
Turns the puzzles from Cho Chikun's problems and turns them into SGFs. Use the Mobile PDFs found [here](https://tsumego.tasuki.org/).

## tsumego_clipping.py

Splits [PDF] -> [separate pages] and [separate pages] -> [individual puzzles].
  
**Usage**: 
  
`pdfToPages(pdfname)`: Converts the PDF at `pdfname` to individual pages in `pageFolder/`

`splitPage(imgname)`: Converts an image for each page of the PDF (`imgname`) and returns an array of cropped images. The cropped images include both puzzles and the text underneath each puzzle.

`saveImgs(imgs, index=1)`: Takes an array of images `imgs`, and saves the ones that are puzzles in `puzzFolder/`. The `index` is used to keep track of how many puzzles have been saved. Returns `index`.

## pngToSGF.py

Converts a png file of a puzzle to a 361 character string denoting stone locations.

`predictStones(i)`: Finds puzzle `i` in the `tsumego/` folder and returns the SGF.

## tsumego.js

Work in progress. See comments at the end of the file. Uses [WGo](http://wgo.waltheri.net/documentation).