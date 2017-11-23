import {
  generateRandomRgbColors,
  getAccessibleColor,
} from './color';

function generateColorSet(m) {
  const rawInputs = generateRandomRgbColors(m);
  const rawTargets = rawInputs.map(getAccessibleColor);

  return { rawInputs, rawTargets };
}

export default generateColorSet;
