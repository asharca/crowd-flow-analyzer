import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts'

interface GenderDistributionChartProps {
  data: Record<string, number>
}

const COLORS: Record<string, string> = {
  male: '#3b82f6',
  female: '#ec4899',
}

function GenderDistributionChart({ data }: GenderDistributionChartProps) {
  const chartData = Object.entries(data).map(([name, value]) => ({
    name,
    value,
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={chartData}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={100}
          paddingAngle={5}
          dataKey="value"
          label={({ name, percent }) =>
            `${name} ${((percent ?? 0) * 100).toFixed(0)}%`
          }
        >
          {chartData.map((entry) => (
            <Cell
              key={entry.name}
              fill={COLORS[entry.name] ?? '#6b7280'}
            />
          ))}
        </Pie>
        <Tooltip formatter={(value) => [String(value), 'People']} />
      </PieChart>
    </ResponsiveContainer>
  )
}

export default GenderDistributionChart
