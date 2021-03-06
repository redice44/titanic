function makeBarGraph(element, data, xField, yField, setWidth, setHeight, label, colorfn) {
  // set the dimensions and margins of the graph
  var margin = {top: 20, right: 20, bottom: 30, left: 40};
  var width = setWidth - margin.left - margin.right;
  var height = setHeight - margin.top - margin.bottom;

  var x = d3.scaleBand()
            .range([0, width])
            .padding(0.1);
  var y = d3.scaleLinear()
            .range([height, 0]);

  var svg = d3.select("#"+ element).append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

  var color = d3.scaleOrdinal(d3.schemeCategory20);
  colorfn = colorfn || function(d, i) { return color(i); };
  console.log(colorfn);

  x.domain(data.map(function(d) { return d[xField]; }));
  y.domain([50, d3.max(data, function(d) { return d[yField]; })]);

  svg.selectAll(".bar")
    .data(data)
  .enter().append("rect")
    .attr("class", "bar")
    .attr("x", function(d) { return x(d[xField]); })
    .attr("width", x.bandwidth())
    .attr("y", function(d) { return y(d[yField]); })
    .attr("height", function(d) { return height - y(d[yField]); })
    .style("fill", colorfn);

  // add the x Axis
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // add the y Axis
  svg.append("g")
    .call(d3.axisLeft(y))
  .append("text")
    .attr("x", -30)
    .attr("y", -10)
    .attr("dy", "0.32em")
    .attr("fill", "#000")
    .attr("font-weight", "bold")
    .attr("text-anchor", "start")
    .text(label);;
}

function makeStackedBarGraph(svg, data, key, field) {
  var margin = {top: 20, right: 20, bottom: 40, left: 50};
  var width = +svg.attr("width") - margin.left - margin.right;
  var height = +svg.attr("height") - margin.top - margin.bottom;
  var xScale = d3.scaleBand().range([0, width]).padding(0.1).align(0.1);
  var yScale = d3.scaleLinear().range([height, 0]);
  var color = d3.scaleOrdinal(['#387bc5', '#c05d56']);
  var xAxis = d3.axisBottom(xScale);
  var yAxis =  d3.axisLeft(yScale);
  var stack = d3.stack()
    .keys(key)
    .order(d3.stackOrderNone)
    .offset(d3.stackOffsetNone);

  var layers= stack(data);
    xScale.domain(data.map(function(d) { return d[field]; }));
    yScale.domain([0, d3.max(layers[layers.length - 1], function(d) { return d[0] + d[1]; }) ]).nice();
  var layer = svg.selectAll(".layer")
    .data(layers)
    .enter().append("g")
    .attr("class", "layer")
    .attr("transform", "translate(40,10)")
    .style("fill", function(d, i) { return color(i); });

  layer.selectAll("rect")
    .data(function(d) { return d; })
  .enter().append("rect")
    .attr("x", function(d) { return xScale(d.data[field]); })
    .attr("y", function(d) { return yScale(d[1]); })
    .attr("height", function(d) { return yScale(d[0]) - yScale(d[1]); })
    .attr("width", xScale.bandwidth());

  svg.append("g")
    .attr("class", "axis axis--x")
    .attr("transform", "translate(30," + (height + 10) + ")")
    .call(xAxis);

  svg.append("g")
    .attr("class", "axis axis--y")
    .attr("transform", "translate(37,10)")
    .call(yAxis)
  .append("text")
    .attr("x", 2)
    .attr("y", yScale(yScale.ticks().pop())+10)
    .attr("dy", "0.32em")
    .attr("fill", "#000")
    .attr("font-weight", "bold")
    .attr("text-anchor", "start")
    .text("Population");


  var legend = svg.append("g")
    .attr("font-family", "sans-serif")
    .attr("font-size", 10)
    .attr("text-anchor", "end")
  .selectAll("g")
  .data(key)
  .enter().append("g")
    .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; });

  legend.append("rect")
    .attr("x", width - 19)
    .attr("width", 19)
    .attr("height", 19)
      .style("fill", function(d, i) { return color(i); });

  legend.append("text")
    .attr("x", width - 24)
    .attr("y", 9.5)
    .attr("dy", "0.32em")
    .text(function(d) { return d; });
}

var dataSets = [
  // {
  //   label: 'All',
  //   key: ['Survived', 'Died'],
  //   data: [
  //     {"All":"female","Survived":233,"Total":314},
  //     {"All":"1st Class","Survived":136,"Total":216},
  //     {"All":"3-Parch","Survived":3,"Total":5},
  //     {"All":"Cherbourg","Survived":93,"Total":168},
  //     {"All":"1-Parch","Survived":65,"Total":118},
  //     {"All":"1-SibSp","Survived":112,"Total":209},
  //     {"All":"2-Parch","Survived":40,"Total":80},
  //     {"All":"2nd Class","Survived":87,"Total":184},
  //     {"All":"2-SibSp","Survived":13,"Total":28},
  //     {"All":"Queenstown","Survived":30,"Total":77},
  //     {"All":"0-SibSp","Survived":210,"Total":608},
  //     {"All":"0-Parch","Survived":233,"Total":678},
  //     {"All":"Southampton","Survived":217,"Total":644},
  //     {"All":"3-SibSp","Survived":4,"Total":16},
  //     {"All":"3rd Class","Survived":119,"Total":491},
  //     {"All":"5-Parch","Survived":1,"Total":5},
  //     {"All":"male","Survived":109,"Total":577},
  //     {"All":"4-SibSp","Survived":3,"Total":18},
  //     {"All":"5-SibSp","Survived":0,"Total":5},
  //     {"All":"4-Parch","Survived":0,"Total":4},
  //     {"All":"6-Parch","Survived":0,"Total":1},
  //     {"All":"8-SibSp","Survived":0,"Total":7}
  //   ],
  // },
  {
    label: 'Sex',
    key: ['Survived', 'Died'],
    data: [
      {"Sex":"Female","Survived":233,"Total":314},
      {"Sex":"Male","Survived":109,"Total":577}
    ]
  },
  {
    label: 'Pclass',
    key: ['Survived', 'Died'],
    data: [
      {"Pclass":"1st Class","Survived":136,"Total":216},
      {"Pclass":"2nd Class","Survived":87,"Total":184},
      {"Pclass":"3rd Class","Survived":119,"Total":491}
    ]
  },
  {
    label: 'SexClass',
    key: ['Survived', 'Died'],
    data: [
      {"SexClass":"F 1st","Survived":91,"Total":94 ,},
      {"SexClass":"F 2nd","Survived":70,"Total":76 ,},
      {"SexClass":"F 3rd","Survived":72,"Total":144,},
      {"SexClass":"M 1st","Survived":45,"Total":122,},
      {"SexClass":"M 2nd","Survived":17,"Total":108,},
      {"SexClass":"M 3rd","Survived":47,"Total":347,}
    ]
  },
  {
    label: 'Age',
    key: ['Survived', 'Died'],
    data: [
      {"Age":0,"Survived":61,"Total":113},
      {"Age":5,"Survived":61,"Total":118},
      {"Age":4,"Survived":40,"Total":103},
      {"Age":1,"Survived":51,"Total":133},
      {"Age":2,"Survived":37,"Total":97},
      {"Age":7,"Survived":38,"Total":101},
      {"Age":6,"Survived":39,"Total":114},
      {"Age":3,"Survived":15,"Total":112}
    ]
  },
  {
    label: 'Fare',
    key: ['Survived', 'Died'],
    data: [
      {"Fare":4,"Survived":113,"Total":176},
      {"Fare":3,"Survived":80,"Total":180},
      {"Fare":2,"Survived":73,"Total":171},
      {"Fare":0,"Survived":52,"Total":241},
      {"Fare":1,"Survived":24,"Total":123}
    ]
  },
  {
    label: 'Family',
    key: ['Survived', 'Died'],
    data: [
      {"Family":4,"Survived":21,"Total":29},
      {"Family":3,"Survived":59,"Total":102},
      {"Family":2,"Survived":89,"Total":161},
      {"Family":7,"Survived":4,"Total":12},
      {"Family":1,"Survived":163,"Total":537},
      {"Family":5,"Survived":3,"Total":15},
      {"Family":6,"Survived":3,"Total":22},
      {"Family":8,"Survived":0,"Total":6},
      {"Family":11,"Survived":0,"Total":7}
    ]
  },
  {
    label: 'Embarked',
    key: ['Survived', 'Died'],
    data: [
      {"Embarked":0,"Survived":93,"Total":168},
      {"Embarked":1,"Survived":30,"Total":77},
      {"Embarked":2,"Survived":219,"Total":646}
    ]
  },
  {
    label: 'Title',
    key: ['Survived', 'Died'],
    data: [
      {"Title":1,"Survived":100,"Total":126},
      {"Title":2,"Survived":130,"Total":185},
      {"Title":3,"Survived":23,"Total":40},
      {"Title":4,"Survived":8,"Total":23},
      {"Title":5,"Survived":81,"Total":517}
    ]
  }
];

dataSets = dataSets.map(function(ds) {
  ds.data = ds.data.map(function(d) {
    d.Died = d.Total - d.Survived;
    return d;
  });
  return ds;
});

// dataSets.forEach(function(d, i) {
//   makeStackedBarGraph(d3.select('#svg' + (i+1)), d.data, d.key, d.label);
// });

var stackedBarGraphData = {
  missingData: [
    {"Feature":"Age", "Found": 714},
    {"Feature":"Cabin", "Found": 204},
    {"Feature":"Embarked", "Found": 889}
  ],
  sexData: [
    {"Sex":"Female","Survived":233,"Total":314},
    {"Sex":"Male","Survived":109,"Total":577}
  ],
  pclassData: [
    {"Pclass":"1st Class","Survived":136,"Total":216},
    {"Pclass":"2nd Class","Survived":87,"Total":184},
    {"Pclass":"3rd Class","Survived":119,"Total":491}
  ],
  sexclassData: [
    {"SexClass":"F 1st","Survived":91,"Total":94 ,},
    {"SexClass":"F 2nd","Survived":70,"Total":76 ,},
    {"SexClass":"F 3rd","Survived":72,"Total":144,},
    {"SexClass":"M 1st","Survived":45,"Total":122,},
    {"SexClass":"M 2nd","Survived":17,"Total":108,},
    {"SexClass":"M 3rd","Survived":47,"Total":347,}
  ],
  ageData: [
    {"Age":0,"Survived":61,"Total":113},
    {"Age":5,"Survived":61,"Total":118},
    {"Age":4,"Survived":40,"Total":103},
    {"Age":1,"Survived":51,"Total":133},
    {"Age":2,"Survived":37,"Total":97},
    {"Age":7,"Survived":38,"Total":101},
    {"Age":6,"Survived":39,"Total":114},
    {"Age":3,"Survived":15,"Total":112}
  ],
  fareData: [
    {"Fare":4,"Survived":113,"Total":176},
    {"Fare":3,"Survived":80,"Total":180},
    {"Fare":2,"Survived":73,"Total":171},
    {"Fare":0,"Survived":52,"Total":241},
    {"Fare":1,"Survived":24,"Total":123}
  ],
  familyData: [
    {"Family":4,"Survived":21,"Total":29},
    {"Family":3,"Survived":59,"Total":102},
    {"Family":2,"Survived":89,"Total":161},
    {"Family":7,"Survived":4,"Total":12},
    {"Family":1,"Survived":163,"Total":537},
    {"Family":5,"Survived":3,"Total":15},
    {"Family":6,"Survived":3,"Total":22},
    {"Family":8,"Survived":0,"Total":6},
    {"Family":11,"Survived":0,"Total":7}
  ],
  embarkedData: [
    {"Embarked":"Cherbourg","Survived":93,"Total":168},
    {"Embarked":"Queenstown","Survived":30,"Total":77},
    {"Embarked":"Southampton","Survived":219,"Total":646}
  ],
  titleData: [
    {"Title":"Mrs","Survived":100,"Total":126},
    {"Title":"Miss","Survived":130,"Total":185},
    {"Title":"Master","Survived":23,"Total":40},
    {"Title":"Other","Survived":8,"Total":23},
    {"Title":"Mr","Survived":81,"Total":517}
  ]
};



var barGraphData = {
  modelData: [
    {"Model":9,"Accuracy":90.57},
    {"Model":10,"Accuracy":90.01},
    {"Model":1,"Accuracy":87.32},
    {"Model":2,"Accuracy":85.86},
    {"Model":3,"Accuracy":85.52},
    {"Model":0,"Accuracy":84.62},
    {"Model":4,"Accuracy":84.62},
    {"Model":13,"Accuracy":83.5},
    {"Model":11,"Accuracy":83.39},
    {"Model":16,"Accuracy":83.28},
    {"Model":14,"Accuracy":81.59},
    {"Model":12,"Accuracy":81.26},
    {"Model":7,"Accuracy":80.36},
    {"Model":5,"Accuracy":79.69},
    {"Model":8,"Accuracy":78.79},
    {"Model":6,"Accuracy":78.0},
    {"Model":15,"Accuracy":61.62}
  ],
  cvData:[
    {"Model":9,"Accuracy":91.607},
    {"Model":10,"Accuracy":91.065},
    {"Model":1,"Accuracy":87.022},
    {"Model":2,"Accuracy":86.008},
    {"Model":3,"Accuracy":85.618},
    {"Model":0,"Accuracy":85.449},
    {"Model":16,"Accuracy":84.688},
    {"Model":4,"Accuracy":84.637},
    {"Model":13,"Accuracy":84.619},
    {"Model":14,"Accuracy":81.981},
    {"Model":12,"Accuracy":81.641},
    {"Model":7,"Accuracy":81.168},
    {"Model":5,"Accuracy":79.762},
    {"Model":8,"Accuracy":77.63},
    {"Model":11,"Accuracy":74.25},
    {"Model":6,"Accuracy":66.584},
    {"Model":15,"Accuracy":62.133}
  ],
  exp1: [
    {"Model":9,"Accuracy":90.57},
    {"Model":10,"Accuracy":90.01},
    {"Model":1,"Accuracy":87.32},
    {"Model":2,"Accuracy":85.86},
    {"Model":3,"Accuracy":85.52},
    {"Model":0,"Accuracy":84.62},
    {"Model":4,"Accuracy":84.62},
    {"Model":13,"Accuracy":83.5},
    {"Model":11,"Accuracy":83.39},
    {"Model":16,"Accuracy":83.28},
    {"Model":14,"Accuracy":81.59},
    {"Model":12,"Accuracy":81.26},
    {"Model":7,"Accuracy":80.36},
    {"Model":5,"Accuracy":79.69},
    {"Model":8,"Accuracy":78.79},
    {"Model":6,"Accuracy":78.0},
    {"Model":15,"Accuracy":61.62}
  ],
  exp2: [
    {"Model":9,"Accuracy":90.57},
    {"Model":10,"Accuracy":89.9},
    {"Model":1,"Accuracy":87.32},
    {"Model":2,"Accuracy":85.86},
    {"Model":3,"Accuracy":85.52},
    {"Model":11,"Accuracy":85.41},
    {"Model":13,"Accuracy":84.96},
    {"Model":4,"Accuracy":84.62},
    {"Model":0,"Accuracy":84.51},
    {"Model":16,"Accuracy":83.28},
    {"Model":15,"Accuracy":82.94},
    {"Model":7,"Accuracy":82.6},
    {"Model":14,"Accuracy":82.27},
    {"Model":12,"Accuracy":81.82},
    {"Model":8,"Accuracy":80.02},
    {"Model":5,"Accuracy":79.24},
    {"Model":6,"Accuracy":70.71}
  ],
  exp3: [
    {"Model":9,"Accuracy":90.57},
    {"Model":10,"Accuracy":90.12},
    {"Model":1,"Accuracy":87.43},
    {"Model":2,"Accuracy":85.86},
    {"Model":3,"Accuracy":85.63},
    {"Model":13,"Accuracy":85.07},
    {"Model":4,"Accuracy":84.74},
    {"Model":0,"Accuracy":84.51},
    {"Model":16,"Accuracy":84.4},
    {"Model":11,"Accuracy":83.95},
    {"Model":7,"Accuracy":82.6},
    {"Model":15,"Accuracy":81.48},
    {"Model":12,"Accuracy":81.26},
    {"Model":8,"Accuracy":80.7},
    {"Model":6,"Accuracy":79.8},
    {"Model":5,"Accuracy":77.33},
    {"Model":14,"Accuracy":61.62}
  ],
  exp4: [
    {"Model":9,"Accuracy":89.79},
    {"Model":10,"Accuracy":89.56},
    {"Model":1,"Accuracy":86.42},
    {"Model":2,"Accuracy":85.97},
    {"Model":3,"Accuracy":84.85},
    {"Model":0,"Accuracy":84.62},
    {"Model":13,"Accuracy":84.62},
    {"Model":4,"Accuracy":83.73},
    {"Model":14,"Accuracy":83.28},
    {"Model":11,"Accuracy":82.38},
    {"Model":16,"Accuracy":82.27},
    {"Model":15,"Accuracy":82.04},
    {"Model":7,"Accuracy":80.02},
    {"Model":6,"Accuracy":79.91},
    {"Model":5,"Accuracy":77.78},
    {"Model":12,"Accuracy":75.31},
    {"Model":8,"Accuracy":74.41}
  ],
  exp5: [
    {"Model":9,"Accuracy":89.34},
    {"Model":10,"Accuracy":89.23},
    {"Model":1,"Accuracy":86.08},
    {"Model":2,"Accuracy":85.07},
    {"Model":3,"Accuracy":84.06},
    {"Model":0,"Accuracy":83.73},
    {"Model":4,"Accuracy":83.39},
    {"Model":13,"Accuracy":83.05},
    {"Model":11,"Accuracy":82.83},
    {"Model":12,"Accuracy":82.49},
    {"Model":15,"Accuracy":82.49},
    {"Model":7,"Accuracy":81.37},
    {"Model":16,"Accuracy":81.26},
    {"Model":5,"Accuracy":80.81},
    {"Model":8,"Accuracy":78.0},
    {"Model":6,"Accuracy":75.08},
    {"Model":14,"Accuracy":61.62}
  ],
  exp6: [
    {"Model":9,"Accuracy":88.44},
    {"Model":10,"Accuracy":87.99},
    {"Model":3,"Accuracy":85.86},
    {"Model":2,"Accuracy":85.19},
    {"Model":0,"Accuracy":84.62},
    {"Model":4,"Accuracy":84.62},
    {"Model":1,"Accuracy":83.95},
    {"Model":13,"Accuracy":83.39},
    {"Model":16,"Accuracy":82.27},
    {"Model":11,"Accuracy":81.93},
    {"Model":6,"Accuracy":78.79},
    {"Model":12,"Accuracy":78.79},
    {"Model":15,"Accuracy":78.79},
    {"Model":7,"Accuracy":78.68},
    {"Model":5,"Accuracy":78.68},
    {"Model":8,"Accuracy":78.68},
    {"Model":14,"Accuracy":61.62}
  ],
  missingData: [
    {"Feature":"Age", "Found": 714},
    {"Feature":"Cabin", "Found": 204},
    {"Feature":"Embarked", "Found": 889}
  ],
  uniqueData: [
    {"Feature":"PassengerId", "Found": 891},
    {"Feature":"Name", "Found": 891},
    {"Feature":"Ticket", "Found": 681},
    {"Feature":"Cabin", "Found": 147}
  ],
  sexData: [
  ]
};

barGraphData.missingData = barGraphData.missingData.map(function(d) {
  d.Missing = 891 - d.Found;
  return d;
});

var survivalData = ['sexData', 'pclassData', 'sexclassData', 'ageData', 'fareData', 'familyData', 'embarkedData', 'titleData'];

survivalData.forEach(function (ds) {
  stackedBarGraphData[ds] = stackedBarGraphData[ds].map(function (d) {
    d.Died = d.Total - d.Survived;
    return d;
  });
});





      // makeStackedBarGraph(d3.select('#svg-missing-data'), stackedBarGraphData.missingData, ['Found', 'Missing'], 'Feature');
