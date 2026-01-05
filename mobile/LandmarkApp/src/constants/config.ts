// API Configuration
// Change this to your backend URL
// For testing on physical device on same WiFi: use your computer's local IP
// Example: export const API_BASE_URL = 'http://192.168.1.100:5126';

export const API_BASE_URL = 'http://172.20.10.2:5126';
// ' http://172.20.10.3:5126'; // choose the one that works for you

export const API_ENDPOINTS = {
  PREDICT: `${API_BASE_URL}/api/prediction/predict`,
  HISTORY: `${API_BASE_URL}/api/prediction/history`,
  HEALTH: `${API_BASE_URL}/api/prediction/health`,
  DATASET_ADD: `${API_BASE_URL}/api/Dataset/add`
};
