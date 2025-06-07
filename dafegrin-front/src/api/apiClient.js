import axios from "axios";

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_BASEURL_SPRING + "/dfg",
  headers: {
    "Content-Type": "application/json",
  },
});

export default apiClient;
