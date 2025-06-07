import React from "react";
import ChartCard from "../components/Chart";
import CorrelationCard from "../components/CorrelationCard";

export default function HomePage() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-8">
      <div className="bg-white border-b shadow-sm py-6 mb-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-extrabold text-gray-800 flex justify-center items-center gap-2">
            <span role="img" aria-label="chart">📊</span>
            DAFEGRIN
          </h1>
          <p className="text-sm text-gray-500">
            DAily FEar&GReed INdex
          </p>
        </div>
      </div>

      <ChartCard />
      <CorrelationCard />
    </div>
  );
}
