/** Compass radians to an ENU unit vector. 0 = north (+Y), π/2 = east (+X). */
export function compassUnit(directionRad: number): { x: number; y: number } {
  return { x: Math.sin(directionRad), y: Math.cos(directionRad) };
}
