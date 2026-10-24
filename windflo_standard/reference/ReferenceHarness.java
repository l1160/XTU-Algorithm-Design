// Local verification harness; the three upstream Java evaluator files are unchanged in logic.
import java.nio.file.*;
import java.util.*;
public class ReferenceHarness {
    public static void main(String[] args) throws Exception {
        WindScenario scenario = new WindScenario(args[0]);
        KusiakLayoutEvaluator evaluator = new KusiakLayoutEvaluator();
        evaluator.initialize(scenario);
        List<String> rows = Files.readAllLines(Paths.get(args[1]));
        double[][] layout = new double[rows.size()][2];
        for (int i=0; i<rows.size(); i++) {
            String[] fields = rows.get(i).trim().split(",");
            layout[i][0] = Double.parseDouble(fields[0]);
            layout[i][1] = Double.parseDouble(fields[1]);
        }
        double cost = evaluator.evaluate(layout);
        System.out.println("RESULT," + evaluator.getEnergyOutput() + "," + evaluator.getWakeFreeRatio() + "," + cost);
        double[][] outputs = evaluator.getEnergyOutputs();
        if (outputs != null) {
            for (int d=0; d<outputs.length; d++) {
                StringBuilder line = new StringBuilder("DIR," + d);
                for (double v : outputs[d]) line.append(",").append(v);
                System.out.println(line.toString());
            }
        }
    }
}
