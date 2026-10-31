import { motion, useReducedMotion } from "motion/react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
export default function OverviewChart({
  chart,
}: {
  chart: { name: string; Accuracy: number; "Macro F1": number }[];
}) {
  const reduced = useReducedMotion();
  return (
    <motion.div className="chart-transition" key={JSON.stringify(chart)} initial={reduced ? false : {opacity:0,clipPath:"inset(0 100% 0 0)"}} animate={{opacity:1,clipPath:"inset(0 0% 0 0)"}} transition={{duration:.4,ease:"easeOut"}}>
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        layout="vertical"
        data={chart}
        barGap={5}
        barCategoryGap="24%"
        margin={{ left: 0, right: 22, top: 8, bottom: 4 }}
      >
        <CartesianGrid
          horizontal={false}
          stroke="var(--border)"
          strokeDasharray="2 4"
        />
        <XAxis
          type="number"
          domain={[0, 100]}
          ticks={[0, 50, 100]}
          tickMargin={12}
          height={40}
          tickFormatter={(v) => v + "%"}
          tick={{ fill: "var(--muted)", fontSize: 16 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={124}
          tickMargin={10}
          tick={{ fill: "var(--muted)", fontSize: 16 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          itemStyle={{ color: "var(--text)" }}
          labelStyle={{ color: "var(--text)" }}
          cursor={{ fill: "var(--surface-subtle)" }}
          formatter={(v) => Number(v).toFixed(2) + "%"}
          contentStyle={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            color: "var(--text)",
            fontSize: 16,
          }}
        />
        <Bar
          dataKey="Accuracy"
          fill="var(--accent)"
          maxBarSize={15}
          isAnimationActive={false}
        />
        <Bar
          dataKey="Macro F1"
          fill="var(--chart-secondary)"
          maxBarSize={15}
          isAnimationActive={false}
        />
      </BarChart>
    </ResponsiveContainer>
    </motion.div>
  );
}
