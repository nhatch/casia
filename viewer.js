
var images = [];


function loadGnt(filename, imageArray, callback) {
  var req = new XMLHttpRequest();
  req.onload = function() {
    populateArray(req.response, imageArray);
    callback();
  }
  req.responseType = "arraybuffer";
  req.open("GET", filename, true);
  req.send();
}

function populateArray(rawGnt, imageArray) {
  var gnt = new Uint8Array(rawGnt);
  constructImages(gnt, imageArray);
}

function shortFromLittleEndianByteArray(bytes) {
  return bytes[0] + bytes[1]*Math.pow(2,8);
}

function shortFromBigEndianByteArray(bytes) {
  return bytes[0]*Math.pow(2,8) + bytes[1];
}

function intFromLittleEndianByteArray(bytes) {
  return bytes[0] + bytes[1]*Math.pow(2,8) + bytes[2]*Math.pow(2,16) + bytes[3]*Math.pow(2,24);
}

function constructImages(gnt, imageArray) {
  var counter = 0;
  while (counter < gnt.length) {
    var length = intFromLittleEndianByteArray(gnt.subarray(counter, counter+4));
    var label = gnt.subarray(counter+4, counter+6);
    var width = shortFromLittleEndianByteArray(gnt.subarray(counter+6, counter+8));
    var height = shortFromLittleEndianByteArray(gnt.subarray(counter+8, counter+10));
    var pixelData = [];
    counter += 10;
    for (var j = 0; j < height; j++) {
      var row = [];
      for (var k = 0; k < width; k++) {
        row.push(gnt[counter++]);
      }
      pixelData.push(row);
    }
    imageArray.push([pixelData, label]);
  }
}

loadGnt("example.gnt", images, function () {
  var imageBrowserDiv = document.createElement("div");
  //var incorrectPredictionsDiv = document.createElement("div");
  createDynamicViewer(imageBrowserDiv, images, 0);
  //createIncorrectPredictionViewers(incorrectPredictionsDiv);
  document.body.appendChild(imageBrowserDiv);
  //document.body.appendChild(incorrectPredictionsDiv);
});

function createDynamicViewer(div, imageArray, startingIndex) {
  var image = imageArray[startingIndex];
  var table = createTable(image);
  var span = document.createElement("span");
  span.className = "container";
  span.appendChild(table);
  span.appendChild(table.label);
  var input = createInputForTable(table, imageArray);
  input.value = startingIndex;
  span.appendChild(input);
  input.focus();
  div.appendChild(span);
}

function createTable(image) {
  var table = document.createElement("table");
  var label = document.createElement("p");
  table.label = label;
  showImage(image, table);
  return table;
}

function createInputForTable(table, imageArray) {
  var input = document.createElement("input");
  input.type = "number";
  input.onchange = function() {
    if (input.value < 0) {
      input.value = 0;
    } else if (input.value >= imageArray.length) {
      input.value = imageArray.length - 1;
    }
    showImage(imageArray[input.value], table);
  }
  return input;
}

function showImage(image, table) {
  table.innerHTML = ""; // remove the old image
  var pixelData = image[0];
  var label = image[1];
  var numRows = pixelData.length;
  var numCols = pixelData[0].length;
  for (var i = 0; i < numRows; i++) {
    var tr = table.insertRow();
    var row = pixelData[i];
    for (var j = 0; j < numCols; j++) {
      var cellString = Number(pixelData[i][j]).toString(16);
      if (cellString.length == 1) { cellString = "0" + cellString; }
      var td = tr.insertCell()
      td.style.backgroundColor = "#" + cellString + cellString + cellString;
    }
  }
  code = Number(label[0]*256 + label[1]).toString(16);
  decodedChar = new TextDecoder("gb18030").decode(label)
  table.label.innerHTML = code + " (" + decodedChar + ")";
}

