import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Linking,
  Image,
  Alert,
} from 'react-native';
import { Card, Button, Portal, Dialog, TextInput } from 'react-native-paper';
import { LandmarkPrediction, PredictionResult, LANDMARK_NAMES, CONFIDENCE_THRESHOLD } from '../types/index';
import { sendPredictionFeedback, uploadDatasetImage } from '../services/api';

interface ResultsScreenProps {
  route: any;
  navigation: any;
}

// Helper to format landmark names nicely (must NOT reference component state)
const formatLandmarkName = (name: string): string => {
  if (LANDMARK_NAMES[name]) return LANDMARK_NAMES[name];

  let decoded = name;
  try {
    decoded = decodeURIComponent(name);
  } catch {
    decoded = name;
  }

  const withSpaces = decoded.replace(/_/g, ' ');

  return withSpaces
    .split(' ')
    .filter(Boolean)
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ');
};


export default function ResultsScreen({ route, navigation }: ResultsScreenProps) {
  const { prediction, imageUri }: { prediction: PredictionResult; imageUri: string } = route.params;

  const topPrediction = prediction.predictions[0];
  const isConfident = topPrediction.confidence >= CONFIDENCE_THRESHOLD;

  const [feedbackVisible, setFeedbackVisible] = useState(false);
  const [correctLabel, setCorrectLabel] = useState('');
  const [comment, setComment] = useState('');
  const [sending, setSending] = useState(false);
  const [datasetUploading, setDatasetUploading] = useState(false);

  const handleWikipediaPress = () => {
    if (topPrediction.wikipediaUrl) Linking.openURL(topPrediction.wikipediaUrl);
  };

  const submitFeedback = async () => {
    try {
      setSending(true);

      await sendPredictionFeedback({
        predictionRecordId: prediction.predictionRecordId ?? null,
        predictedLabel: topPrediction.label,
        predictedConfidence: topPrediction.confidence,
        correctLabel: correctLabel.trim() || null,
        comment: comment.trim() || null,
      });

      setFeedbackVisible(false);
      setCorrectLabel('');
      setComment('');
      Alert.alert('Thank you', 'Your feedback was sent.');
    } catch (e: any) {
      Alert.alert('Error', e?.message ?? 'Failed to send feedback.');
    } finally {
      setSending(false);
    }
  };

  const addToDatasetAsPredicted = async () => {
    if (!imageUri) return;

    try {
      setDatasetUploading(true);
      await uploadDatasetImage(topPrediction.label, imageUri);
      Alert.alert('Saved', `Added to: ${topPrediction.label}`);
    } catch (e: any) {
      Alert.alert('Upload failed', e?.message ?? 'Unknown error');
    } finally {
      setDatasetUploading(false);
    }
  };

  return (
    <>
      <ScrollView style={styles.container}>
        {/* Image */}
        {imageUri && (
          <Card style={styles.imageCard}>
            <Image source={{ uri: imageUri }} style={styles.image} />
          </Card>
        )}

        {/* Confidence warning */}
        {!isConfident && (
          <Card style={styles.warningCard}>
            <Card.Content>
              <Text style={styles.warningTitle}>⚠️ Low Confidence</Text>
              <Text style={styles.warningText}>
                The model is not very confident about this prediction.
                Try taking another photo with better lighting or angle.
              </Text>
            </Card.Content>
          </Card>
        )}

        {/* Top prediction */}
        <Card style={styles.topPredictionCard}>
          <Card.Content>
            <Text style={styles.topLabel}>Top Prediction</Text>
            <Text style={styles.topLandmark}>
              {formatLandmarkName(topPrediction.label)}
            </Text>
            <Text style={styles.topConfidence}>
              {(topPrediction.confidence * 100).toFixed(1)}% confident
            </Text>

            {/* Wikipedia Button */}
            {topPrediction.wikipediaUrl ? (
              <TouchableOpacity
                style={styles.wikiButton}
                onPress={handleWikipediaPress}
              >
                <Text style={styles.wikiButtonText}>📖 Read on Wikipedia</Text>
              </TouchableOpacity>
            ) : (
              <Text style={styles.noWiki}>Wikipedia article not available</Text>
            )}

            {/* Report wrong prediction */}
            <View style={{ marginTop: 12 }}>
              <Button
                mode="outlined"
                onPress={() => setFeedbackVisible(true)}
                textColor="#d32f2f"
                style={{ borderColor: '#d32f2f' }}
              >
                Report wrong prediction
              </Button>
            </View>
             <View style={{ marginTop: 12 }}>
              <Button
                  mode="outlined"
                  textColor="white"
                  onPress={addToDatasetAsPredicted}
                  disabled={!imageUri || datasetUploading}
                  loading={datasetUploading}
                  >
                  Add photo to dataset (predicted label)
                </Button>     
            </View>
          </Card.Content>
        </Card>


        {/* All predictions */}
        <Card style={styles.predictionsCard}>
          <Card.Content>
            <Text style={styles.sectionTitle}>Top 3 Predictions</Text>

            {prediction.predictions.map((pred: LandmarkPrediction, index: number) => (
              <View key={index} style={styles.predictionItem}>
                <View style={styles.predictionHeader}>
                  <Text style={styles.predictionRank}>#{pred.rank}</Text>
                  <Text style={styles.predictionLabel}>
                    {formatLandmarkName(pred.label)}
                  </Text>
                  <Text style={styles.predictionConfidence}>
                    {(pred.confidence * 100).toFixed(1)}%
                  </Text>
                </View>
                <View style={styles.progressBar}>
                  <View
                    style={[
                      styles.progressFill,
                      {
                        width: `${pred.confidence * 100}%`,
                        backgroundColor: pred.rank === 1 ? '#6200ea' : '#9e9e9e',
                      },
                    ]}
                  />
                </View>
              </View>
            ))}
          </Card.Content>
        </Card>

        {/* Inference time */}
        <Card style={styles.infoCard}>
          <Card.Content>
            <Text style={styles.infoText}>
              ⚡ Inference time: {prediction.inferenceTimeMs}ms
            </Text>
          </Card.Content>
        </Card>

        {/* Actions */}
        <View style={styles.buttonContainer}>
          <Button
            mode="contained"
            onPress={() => navigation.navigate('Camera')}
            style={styles.button}
          >
            Take Another Photo
          </Button>

          <Button
            mode="outlined"
            onPress={() => navigation.navigate('History')}
            style={styles.button}
          >
            View History
          </Button>
        </View>
      </ScrollView>

      {/* Feedback dialog */}
      <Portal>
        <Dialog visible={feedbackVisible} onDismiss={() => !sending && setFeedbackVisible(false)}>
          <Dialog.Title >Report wrong prediction</Dialog.Title>
          <Dialog.Content>
            <Text style={{ marginBottom: 8 }}>
              Predicted: {formatLandmarkName(topPrediction.label)}
            </Text>

            <TextInput
              label="Correct landmark (optional)"
              value={correctLabel}
              onChangeText={setCorrectLabel}
              autoCapitalize="none"
              style={{ marginBottom: 12 }}
            />

            <TextInput
              label="Comment (optional)"
              value={comment}
              onChangeText={setComment}
              multiline
            />
          </Dialog.Content>
          <Dialog.Actions>
            <Button onPress={() => setFeedbackVisible(false)} disabled={sending}>
              Cancel
            </Button>
            <Button onPress={submitFeedback} loading={sending} disabled={sending}>
              Send
            </Button>
          </Dialog.Actions>
        </Dialog>
      </Portal>
    </>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  imageCard: {
    margin: 15,
    elevation: 4,
  },
  image: {
    width: '100%',
    height: 250,
    borderRadius: 8,
  },
  warningCard: {
    margin: 15,
    marginTop: 0,
    backgroundColor: '#fff3cd',
    elevation: 2,
  },
  warningTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#856404',
    marginBottom: 5,
  },
  warningText: {
    fontSize: 14,
    color: '#856404',
  },
  topPredictionCard: {
    margin: 15,
    marginTop: 0,
    backgroundColor: '#6200ea',
    elevation: 4,
  },
  topLabel: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
    marginBottom: 5,
  },
  topLandmark: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 5,
  },
  topConfidence: {
    fontSize: 16,
    color: '#fff',
    opacity: 0.9,
    marginBottom: 15,
  },
  wikiButton: {
    backgroundColor: '#fff',
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 10,
  },
  wikiButtonText: {
    color: '#6200ea',
    fontSize: 14,
    fontWeight: '600',
  },
  noWiki: {
    fontSize: 12,
    color: 'rgba(255,255,255,0.7)',
    fontStyle: 'italic',
    marginTop: 10,
    textAlign: 'center',
  },
  predictionsCard: {
    margin: 15,
    marginTop: 0,
    elevation: 2,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#333',
  },
  predictionItem: {
    marginBottom: 20,
  },
  predictionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  predictionRank: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#6200ea',
    marginRight: 10,
  },
  predictionLabel: {
    flex: 1,
    fontSize: 16,
    color: '#333',
  },
  predictionConfidence: {
    fontSize: 14,
    color: '#666',
    fontWeight: '600',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
    backgroundColor: '#e0e0e0',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 4,
  },
  infoCard: {
    margin: 15,
    marginTop: 0,
    elevation: 1,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
  },
  buttonContainer: {
    margin: 15,
    marginTop: 0,
    gap: 10,
    marginBottom: 30,
  },
  button: {
    paddingVertical: 8,
  },
});