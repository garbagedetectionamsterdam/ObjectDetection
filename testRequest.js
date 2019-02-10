var fs = require('fs')
var request = require('request');

function requestPrediction(apiUrl, imageFilePath, callback)
{

	var rs = fs.createReadStream(imageFilePath);
	var ws = request.post(apiUrl, function(error, response, body) {
		error && console.log(error)
		callback(body)

	});

	ws.on('drain', function () {
		console.log('drain', new Date());
		rs.resume();
	});

	rs.on('end', function () {
		console.log('uploaded to ' + apiUrl);
	});

	ws.on('error', function (err) {
		console.error('cannot send file to ' + apiUrl + ': ' + err);
	});
	console.log('commencing file pipe stream')

	rs.pipe(ws);
	console.log('returning control to event queue')

}


console.log(process.argv);
var apiTarget = process.argv[2];
var filePath = process.argv[3];
console.log(apiTarget);
console.log(filePath);
var startTime = new Date().getTime();
requestPrediction(apiTarget, filePath, body => {
	var endTime = new Date().getTime();
	console.log('total time cost:');
	console.log(endTime - startTime);

	console.log('body:');
	console.log(body);

})
