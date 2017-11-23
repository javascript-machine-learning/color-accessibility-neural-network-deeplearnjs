// Returns either black or white as output rgb color depending on input rgb color
// Can be used to display font in a more accessbile way on a colored background
function getAccessibleColor(rgb) {
  let [ r, g, b ] = rgb;

  let uicolors = [r / 255, g / 255, b / 255];

  let c = uicolors.map((col) => {
    if (col <= 0.03928) {
      return col / 12.92;
    }
    return Math.pow((col + 0.055) / 1.055, 2.4);
  });

  let L = (0.2126 * c[0]) + (0.7152 * c[1]) + (0.0722 * c[2]);

  return (L > 0.179)
    ? [ 0, 1 ] // black
    : [ 1, 0 ]; // white
}

function generateRandomRgbColors(m) {
  const rawInputs = [];

  for (let i = 0; i < m; i++) {
    rawInputs.push(generateRandomRgbColor());
  }

  return rawInputs;
}

function generateRandomRgbColor() {
  return [
    randomIntFromInterval(0, 255),
    randomIntFromInterval(0, 255),
    randomIntFromInterval(0, 255),
  ];
}

function randomIntFromInterval(min, max) {
  return Math.floor(Math.random() * (max - min + 1) + min);
}

export {
  generateRandomRgbColors,
  getAccessibleColor,
};