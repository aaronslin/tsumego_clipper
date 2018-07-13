BOARD_SIZE = 19
EMPTY_CHAR = "."
BLACK_CHAR = "b"
WHITE_CHAR = "w"



function parseRawSGF(raw) {
	sgf = []
	for(var i=0; i<BOARD_SIZE; i=i+1) {
		start = BOARD_SIZE*i;
		end = BOARD_SIZE*(i+1);
		sgf.push(Array.from(raw.slice(start, end)))
	}
	return sgf
}

function getInitPosition(sgf, colorMap) {
	objects = []
	for(var i=0; i < BOARD_SIZE; i=i+1) {
		for(var j=0; j < BOARD_SIZE; j=j+1) {
			var c = sgf[i][j]
			if (c==EMPTY_CHAR) {
				continue
			}
			stone = {
				x: j, 
				y: i, 
				c: colorMap[c]
			}
			objects.push(stone)
		}
	}
	return objects
}

function getColorMap() {
	colors = [WGo.B, WGo.W]
	rand = Math.floor(Math.random()*2)
	colorMap = {[BLACK_CHAR]: colors[rand], [WHITE_CHAR]: colors[1-rand]};
	return colorMap
}

function getSection(sgf, margin) {
	topV = 0;
	left = 0;
	right = 0;
	bottom = 0;

	// bottom
	function isEmpty(x) {
		return x==EMPTY_CHAR;
	}
	for(var i=1; i<BOARD_SIZE+1; i=i+1) {
		if (sgf[BOARD_SIZE-i].every(isEmpty)) {
			bottom+=1;
		}
		else {
			break
		}
	}
	// top
	for(var i=0; i<BOARD_SIZE; i=i+1) {
		if (sgf[i].every(isEmpty)) {
			topV+=1;
		}
		else {
			break
		}
	}

	// right 
	function buildEmptyCol(i) {
		emptyCol = function(x) {
			return x[i] == EMPTY_CHAR
		}
		return emptyCol
	}
	for(var i=1; i<BOARD_SIZE+1; i=i+1) {
		if (sgf.every(buildEmptyCol(BOARD_SIZE-i))) {
			right+=1
		}
		else {
			break
		}
	}
	// left
	for(var i=0; i<BOARD_SIZE; i=i+1) {
		if (sgf.every(buildEmptyCol(i))) {
			left+=1
		}
		else {
			break
		}
	}



	// Margins
	topV = Math.max(0, topV-margin)
	left = Math.max(0, left-margin)
	bottom = Math.max(0,bottom-margin)
	right = Math.max(0, right-margin)

	return {top: topV, left:left, right:right, bottom: bottom}
}

function randomizeOrientation(sgf) {
	// Rotations
	numRotations = Math.floor(4*Math.random())
	function rotateIndices(indices) {
		return [BOARD_SIZE-1-indices[1], indices[0]]
	}
	for(var r=0; r<numRotations; r=r+1) {
		for(var i=0; i<Math.floor(BOARD_SIZE/2); i=i+1) {
			for(var j=0; j<Math.ceil(BOARD_SIZE/2); j=j+1) {
				temp = sgf[i][j]
				X = [i, j];
				for(var k=0; k<3; k=k+1) {
					Y = rotateIndices(X)
					sgf[X[0]][X[1]] = sgf[Y[0]][Y[1]]
					X = Y;
				}
				sgf[X[0]][X[1]] = temp;
			}
		}
	}
	
	// Transposing
	numFlips = Math.floor(2*Math.random())
	if(numFlips) {
		for(var i=0; i<BOARD_SIZE; i=i+1) {
			for(var j=0; j<i; j=j+1) {
				temp = sgf[i][j]
				sgf[i][j] = sgf[j][i]
				sgf[j][i] = temp
			}
		}
	}
	return sgf
}

sgf = ".....wwb...........wwb.w.wb...........bbwwwwb..............bbbbb.............b..........................................................................................................................................................................................................................................................................................."


TO_PLAY = BLACK_CHAR;

colorMap = getColorMap();
parsed = parseRawSGF(sgf)
parsed = randomizeOrientation(sgf)
pos = getInitPosition(sgf, colorMap)

var board = new WGo.Board(document.getElementById("board"), {
    width: 600,
	section: getSection(parsed, 0)
});

board.addObject(pos)



