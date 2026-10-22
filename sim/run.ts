import { parseSeed } from "./args.ts";
import { runScenario } from "./scenario.ts";

const seed = parseSeed(process.argv.slice(2));
const result = runScenario({ seed });
console.log(`seed ${result.seed}`);
console.log(`landed ${result.landed ? "yes" : "no"}`);
console.log(`steps ${result.steps}`);
console.log(`duration_s ${result.durationS.toFixed(2)}`);
console.log(`final_error_m ${result.finalErrorM.toFixed(2)}`);
console.log(`final_alt_m ${result.finalAltM.toFixed(2)}`);
process.exit(result.landed ? 0 : 1);
