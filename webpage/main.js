function getPuzzle(id) {
	return allPuzzles[id];
}

var boardElement = document.querySelector(".tenuki-board");
var game = new tenuki.Game(boardElement);
game.setup({
	startPosition: getPuzzle(2)
});

var controlElement = document.querySelector(".controls");
var controls = new ExampleGameControls(controlElement, game);
controls.setup();

game.callbacks.postRender = function(game) {
	controls.updateStats();
};