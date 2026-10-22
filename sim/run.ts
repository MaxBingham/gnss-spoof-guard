import { parseSeed } from "./args.ts";
import { runScenario } from "./scenario.ts";

const seed = parseSeed(process.argv.slice(2));
const result = runScenario({ seed });
const last = result.samples.at(-1);
const estimateErrorM = last
  ? Math.hypot(last.trace.estimate.x - last.truth.x, last.trace.estimate.y - last.truth.y)
  : Number.NaN;
console.log(`seed ${result.seed}`);
console.log(`landed ${result.landed ? "yes" : "no"}`);
console.log(`steps ${result.steps}`);
console.log(`duration_s ${result.durationS.toFixed(2)}`);
console.log(`final_error_m ${result.finalErrorM.toFixed(2)}`);
console.log(`final_alt_m ${result.finalAltM.toFixed(2)}`);
console.log(`samples ${result.samples.length}`);
if (last) {
  console.log(`truth_m ${last.truth.x.toFixed(2)} ${last.truth.y.toFixed(2)} ${last.truth.z.toFixed(2)}`);
  console.log(
    `estimate_m ${last.trace.estimate.x.toFixed(2)} ${last.trace.estimate.y.toFixed(2)} ${last.trace.estimate.z.toFixed(2)}`,
  );
  console.log(`estimate_error_m ${estimateErrorM.toFixed(2)}`);
}
process.exit(result.landed ? 0 : 1);
