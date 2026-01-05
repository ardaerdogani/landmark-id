// TypeScript types for the Landmark Identification app

export interface LandmarkPrediction {
  label: string;
  confidence: number;
  rank: number;
  wikipediaUrl?: string;
}

export interface PredictionResult {
  predictions: LandmarkPrediction[];
  inferenceTimeMs: number;
  predictionRecordId?: number | null;
}

export interface HistoryRecord {
  id: number;
  originalFilename: string;
  timestamp: string;
  topPrediction: string;
  topConfidence: number;
  inferenceTimeMs: number;
  imageSizeMb: number;
}

export const LANDMARK_NAMES: Record<string, string> = {
  'Trakai_Island_Castle': 'Trakai Island Castle',
  'Rasos_Cemetery': 'Rasos Cemetery',
  'Hill_of_Crosses': 'Hill of Crosses',
  'IX_Fort': 'Ninth Fort',
  'Gediminas_Tower': 'Gediminas Tower',
  'Gate_of_Dawn': 'Gate of Dawn',
  'Uzupis': 'Užupis',
  'Parnid%C5%BEio_kopa': 'Parnidis Dune',
};

export const CONFIDENCE_THRESHOLD = 0.55; // Below this is "unsure"
