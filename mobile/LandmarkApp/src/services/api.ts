import axios from 'axios';
import { API_ENDPOINTS } from '../constants/config';
import { PredictionResult, HistoryRecord } from '../types';
import { API_BASE_URL } from '../constants/config';


// Create axios instance with timeout
const apiClient = axios.create({
  timeout: 30000, // 30 seconds
});

/**
 * Upload an image to the prediction API
 * @param imageUri - Local file URI from camera/gallery
 * @returns Prediction result with Top-3 predictions
 */

export const predictLandmark = async (imageUri: string): Promise<PredictionResult> => {
  const form = new FormData();

  const name = imageUri.split('/').pop() || 'photo.jpg';
  const lower = name.toLowerCase();
  const type = lower.endsWith('.png') ? 'image/png' : 'image/jpeg';

  form.append('imageFile', { uri: imageUri, name, type } as any);

  const res = await apiClient.post(API_ENDPOINTS.PREDICT, form, {
    headers: { Accept: 'application/json' },
  });

  return res.data;
};

/**
 * Get prediction history from the server
 * @param limit - Number of records to fetch (default: 20)
 * @returns Array of history records
 */
export const getHistory = async (limit: number = 20): Promise<HistoryRecord[]> => {
  const response = await apiClient.get<HistoryRecord[]>(
    `${API_ENDPOINTS.HISTORY}?limit=${limit}`
  );

  return response.data;
};

/**
 * Check if the API is healthy
 * @returns true if API is reachable and healthy
 */
export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await apiClient.get(API_ENDPOINTS.HEALTH, {
      timeout: 5000,
    });
    return response.status === 200;
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};

export async function sendPredictionFeedback(payload: {
  predictionRecordId?: number | null;
  predictedLabel?: string | null;
  predictedConfidence?: number | null;
  correctLabel?: string | null;
  comment?: string | null;
}) {
  const res = await fetch(`${API_BASE_URL}/api/Feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Feedback failed (${res.status}): ${text}`);
  }

  return res.json();
}

export async function uploadDatasetImage(label: string, imageUri: string) {
  const form = new FormData();
  form.append('label', label);

  form.append('imageFile', {
    uri: imageUri,
    name: 'photo.jpg',
    type: 'image/jpeg',
  } as any);

  const res = await fetch(API_ENDPOINTS.DATASET_ADD, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }

  return res.json();
}