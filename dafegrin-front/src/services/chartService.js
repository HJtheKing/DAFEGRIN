import apiClient from "../api/apiClient";

export const fetchDailyIndexes = () => {
  return apiClient.get("/idx");
};
