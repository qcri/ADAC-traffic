const fs = require('fs');

function renameFiles(path) {
  var files = fs.readdirSync(path);
  for (var i in files) {
    var file = files[i];
    if (file.startsWith('osm')) {
      var newFile = file.replace('osm', 'ec');
      fs.renameSync(path + file, path + newFile);
    }
  }
}

renameFiles('./');