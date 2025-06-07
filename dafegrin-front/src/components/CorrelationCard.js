import React, { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogTitle } from "./ui/dialog";
import { fetchDailyIndexes } from "../services/chartService";
import axios from "axios";
import {
  ScatterChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Scatter,
} from "recharts";

export default function CorrelationCard() {
  const [correlationData, setCorrelationData] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [scatterData, setScatterData] = useState([]);

  // 1. 피어슨 상관계수 API 호출
  useEffect(() => {
    axios
      .get(process.env.REACT_APP_BASEURL_FASTAPI + "/analyze")
      .then((res) => setCorrelationData(res.data))
      .catch((err) => console.error("상관계수 API 오류:", err));
  }, []);

  // 2. 모달 열릴 때 scatterData 세팅
  const openModal = () => {
    fetchDailyIndexes()
      .then((res) => {
        const data = res.data.map((item) => ({
          kospi: item.kospi,
          aIndex: item.sentiment,
        }));
        setScatterData(data);
        setModalOpen(true);
      })
      .catch((err) => console.error("차트 데이터 호출 실패:", err));
  };

  if (!correlationData) return null;

  return (
    <div className="bg-white shadow-md rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold text-gray-700">
          📈 피어슨 상관분석
        </h3>
        <span className="text-sm text-gray-500">
          (클릭 시 산점도 보기)
        </span>
      </div>

      <div
        className="cursor-pointer p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition"
        onClick={openModal}
      >
        <p className="text-gray-600">
          상관계수:{" "}
          <span className="font-bold text-blue-600">
            {correlationData.correlation.toFixed(3)}
          </span>
        </p>
        <p className="text-gray-600">
          유의확률:{" "}
          <span className="font-semibold text-indigo-600">
            {correlationData.p_value.toFixed(4)}
          </span>
        </p>
        <p className="text-gray-600">
          데이터 수:{" "}
          <span className="font-semibold text-gray-700">
            {correlationData.count}
          </span>
        </p>
      </div>

      <Dialog open={modalOpen} onOpenChange={setModalOpen}>
        <DialogContent>
          <div className="flex items-center justify-between mb-4">
            <DialogTitle className="text-lg font-semibold text-gray-700">
              KOSPI & DAFEGRIN 산점도
            </DialogTitle>
            <button
              className="text-gray-400 hover:text-gray-700"
              onClick={() => setModalOpen(false)}
            >
              ✕
            </button>
          </div>

          <div className="flex justify-center overflow-x-auto">
            <ScatterChart
              width={600}
              height={400}
              margin={{ top: 20, right: 20, bottom: 30, left: 0 }}
            >
              <CartesianGrid stroke="#e5e7eb" />
              <XAxis
                type="number"
                dataKey="aIndex"
                name="감정 점수"
                label={{ value: "DAFEGRIN", position: "insideBottom", offset: -5, dy: 15 }}
                axisLine={{ stroke: "#cbd5e1" }}
                tickLine={false}
                tick={{ fill: "#475569" }}
              />
              <YAxis
                type="number"
                dataKey="kospi"
                name="KOSPI 변동률"
                label={{ value: "KOSPI", angle: -90, position: "insideLeft", offset: -10, dx: 25, style: {textAnchor: "middle"}}}
                axisLine={{ stroke: "#cbd5e1" }}
                tickLine={false}
                tick={{ fill: "#475569" }}
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                contentStyle={{ fontSize: "0.875rem" }}
              />
              <Scatter name="지표" data={scatterData} fill="#3b82f6" />
            </ScatterChart>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
