/**
 * API Service Client connecting to FastAPI backend.
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000,
});

export const fetchRegionalBasins = async () => {
  const response = await apiClient.get('/regional-basins');
  return response.data;
};

export const fetchLiveStorms = async () => {
  const response = await apiClient.get('/live-storms');
  return response.data;
};

export const runFullCyclonePipeline = async (payload) => {
  const response = await apiClient.post('/full-pipeline', payload);
  return response.data;
};

export default apiClient;
