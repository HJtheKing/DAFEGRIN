import React, { useEffect, useState } from "react";
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
} from "recharts";
import { Dialog, DialogContent, DialogTitle } from "./ui/dialog";
import { fetchDailyIndexes } from "../services/chartService";
import { fetchNewsByDailyIndexId } from "../services/newsService";

export default function ChartCard() {
  const [chartData, setChartData] = useState([]);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedData, setSelectedData] = useState(null);
  const [newsData, setNewsData] = useState([]);

  useEffect(() => {
    fetchDailyIndexes()
      .then((res) => {
        const processed = res.data.map((item) => {
          const aIndex = item.sentiment;
          const centered = parseFloat((aIndex - 0.5).toFixed(2));
          return {
            id: item.id,
            date: item.date,
            kospi: item.kospi,
            aIndex,
            aIndexCentered: centered,
            aIndexColor: centered >= 0 ? "#ef4444" : "#3b82f6",
          };
        });
        setChartData(processed);
      })
      .catch((err) => {
        console.error("API 오류:", err);
      });
  }, []);

  const handleClick = (e) => {
    const clickedDate = e.activeLabel;
    const data = chartData.find((d) => d.date === clickedDate);
    if (data) {
      setSelectedData(data);
      fetchNewsByDailyIndexId(data.id)
        .then((res) => {
          setNewsData(res.data);
          setModalOpen(true);
        })
        .catch((err) => {
          console.error("뉴스 API 오류:", err);
        });
    }
  };

  const chartWidth = Math.max(chartData.length * 100, 800);

  return (
    <div className="bg-white shadow-md rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-700">
          KOSPI & DAFEGRIN
        </h2>
        <span className="text-sm text-gray-500">(클릭 시 뉴스 보기)</span>
      </div>

      <div className="flex justify-center overflow-x-auto">
        <div
          className="min-w-[800px] bg-gray-50 rounded-md"
          style={{ width: `${chartWidth}px`, height: 300 }}
        >
          <ComposedChart
            width={chartWidth}
            height={300}
            data={chartData}
            onClick={handleClick}
          >
            <CartesianGrid stroke="#e5e7eb" />
            <XAxis
              dataKey="date"
              axisLine={{ stroke: "#cbd5e1" }}
              tickLine={false}
              tick={{ fontSize: 12, fill: "#475569" }}
            />
            <YAxis
              yAxisId="left"
              domain={[-5, 5]}
              tickFormatter={(v) => `${v}%`}
              tick={false}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[-0.5, 0.5]}
              ticks={[-0.5, -0.25, 0, 0.25, 0.5]}
              tick={false}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              content={({ active, payload, label }) => {
                if (!active || !payload || payload.length === 0) return null;
                const data = payload[0].payload;
                return (
                  <div className="bg-white p-2 border rounded shadow text-sm">
                    <p className="font-semibold mb-1">날짜: {label}</p>
                    <p className="text-red-500">감정 지수: {data.aIndex}</p>
                    <p className="text-blue-500">
                        코스피 변동률: {data.kospi != null ? `${data.kospi}%` : "-"}
                    </p>
                  </div>
                );
              }}
            />
            <ReferenceLine
              yAxisId="right"
              y={0}
              stroke="#94a3b8"
              strokeDasharray="3 3"
              strokeWidth={1}
            />
            <Bar yAxisId="right" dataKey="aIndexCentered" barSize={40} fill="#ccc">
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.aIndexColor} />
              ))}
            </Bar>
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="kospi"
              stroke="#1e40af"
              strokeWidth={2}
              dot={{ r: 3, fill: "#1e40af" }}
            />
          </ComposedChart>
        </div>
      </div>

      <Dialog open={modalOpen} onOpenChange={setModalOpen}>
        <DialogContent>
          <div className="flex items-center justify-between mb-4">
            <DialogTitle className="text-lg font-semibold text-gray-700">
              {selectedData?.date?.slice(0, 10)} 뉴스 목록
            </DialogTitle>
            <button
              className="text-gray-400 hover:text-gray-700"
              onClick={() => setModalOpen(false)}
            >
              ✕
            </button>
          </div>

          {newsData.length > 0 ? (
            <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
              <table className="w-full text-sm border border-gray-200">
                <thead className="bg-gray-100 sticky top-0 z-10">
                  <tr>
                    <th className="border px-3 py-2 text-center font-medium text-gray-600">
                      날짜
                    </th>
                    <th className="border px-3 py-2 text-center font-medium text-gray-600">
                      뉴스 내용
                    </th>
                    <th className="border px-3 py-2 text-center font-medium text-gray-600">
                      감정 점수
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {newsData.map((news, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="border px-3 py-2 text-center text-gray-700">
                        {news.date?.slice(0, 10)}
                      </td>
                      <td className="border px-3 py-2 text-gray-700">
                        {news.content.length > 50
                          ? news.content.slice(0, 50) + "… (생략)"
                          : news.content}
                      </td>
                      <td className="border px-3 py-2 text-center text-gray-700">
                        {Number(news.posProb).toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-center text-gray-500 mt-4">뉴스가 없습니다.</p>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
