import * as d3 from 'd3';

const directoryPath = 'simulation_runs';
const timeSeriesData = []

const radius = 10;
const xOffset = radius * 0.5;
const yOffset = radius * Math.sqrt(3) / 2;

function generateHexGrid(width, height) {
  const points = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const point = {
        x: x * (radius + xOffset) + y % 2,
        y: y * yOffset
      };
      points.push(point);
    }

    isOffsetRow = !isOffsetRow;
  }

  return points;
}

function main() {
  const configPath = path.join(__dirname, 'simulation_runs/20230915_12:15:33_tensors/config.yaml');
  const configData = fs.readFileSync(configPath, 'utf8');
  const config = yaml.safeLoad(configData);

  const width = config.environment.width;
  const height = config.environment.height;

  points = generateHexGrid(width, heigh)
  

  // Draw points on a screen
  const svg = d3.select("body")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  svg.selectAll("circle")
    .data(points)
    .enter()
    .append("circle")
    .attr("cx", (d) => d.x)
    .attr("cy", (d) => d.y)
    .attr("r", radius);
  
}

main()