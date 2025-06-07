import apiClient from "../api/apiClient";

export const fetchNewsByDailyIndexId = (dailyIndexId) => {
  return apiClient.get(`/news/${dailyIndexId}`);
};
