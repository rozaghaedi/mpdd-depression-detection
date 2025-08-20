import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json
import os

class MPDDDataset(Dataset):
    """
    Custom PyTorch Dataset for MPDD (Multimodal Personalized Depression Detection)

    This dataset handles:
    - Audio features (multiple formats: mfcc, wav2vec, opensmile)
    - Visual features (multiple formats: resnet, densenet, openface)
    - Personalized features (text embeddings)
    - Multi-class labels (binary, ternary, quinary)
    """

    def __init__(self,
                 data_path,
                 window_size='5s',
                 audio_feature_type='wav2vec',
                 visual_feature_type='resnet',
                 task_type='binary',
                 split='train',
                 config=None,
                 transform=None):

        self.data_path = data_path
        self.window_size = window_size
        self.audio_feature_type = audio_feature_type
        self.visual_feature_type = visual_feature_type
        self.task_type = task_type
        self.split = split
        self.config = config
        self.transform = transform

        # Load data labels
        self.labels_file = os.path.join(data_path, 'Training', 'labels', 'Training_Validation_files.json')
        with open(self.labels_file, 'r') as f:
            self.data_list = json.load(f)

        # Load personalized features (embeddings)
        self.personalized_file = os.path.join(data_path, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')
        personalized_data = np.load(self.personalized_file, allow_pickle=True)

        # Create personalized features mapping
        self.personalized_features = {}
        if isinstance(personalized_data, np.ndarray) and len(personalized_data) > 0:
            if isinstance(personalized_data[0], dict):
                # Format: [{'id': '1', 'embedding': [...]}, ...]
                for item in personalized_data:
                    self.personalized_features[str(item['id'])] = item['embedding']
            else:
                # Alternative format handling
                print("⚠️  Unexpected personalized features format, using fallback")

        # Set feature paths
        self.audio_path = os.path.join(data_path, 'Training', window_size, 'Audio', audio_feature_type)
        self.visual_path = os.path.join(data_path, 'Training', window_size, 'Visual', visual_feature_type)

        # Task type mapping
        self.task_mapping = {
            'binary': 'bin_category',
            'ternary': 'tri_category',
            'quinary': 'pen_category'
        }

        # Filter out samples with missing files
        self._filter_valid_samples()

        print(f"Loaded {len(self.data_list)} samples for {task_type} classification")
        print(f"Audio features from: {self.audio_path}")
        print(f"Visual features from: {self.visual_path}")

    def _filter_valid_samples(self):
        """Filter out samples where either audio or visual files are missing"""
        valid_samples = []
        missing_count = 0

        for item in self.data_list:
            audio_file = os.path.join(self.audio_path, item['audio_feature_path'])
            visual_file = os.path.join(self.visual_path, item['video_feature_path'])

            if os.path.exists(audio_file) and os.path.exists(visual_file):
                valid_samples.append(item)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"⚠️  Filtered out {missing_count} samples with missing files")
            print(f"📊  Valid samples: {len(valid_samples)}/{len(self.data_list)}")

        self.data_list = valid_samples

        # Additional check: verify that at least some samples have the required task label
        if len(self.data_list) > 0:
            task_label_key = self.task_mapping[self.task_type]
            samples_with_label = sum(1 for item in self.data_list if task_label_key in item)
            if samples_with_label == 0:
                print(f"⚠️  No samples have {task_label_key} labels. Task '{self.task_type}' may not be supported for this dataset.")
            else:
                print(f"✅  Found {samples_with_label}/{len(self.data_list)} samples with {task_label_key} labels")

    def __len__(self):
        return len(self.data_list)

    def pad_or_truncate(self, feature, max_len):
        """Pad or truncate feature sequence to fixed length"""
        if isinstance(feature, np.ndarray):
            feature = torch.tensor(feature, dtype=torch.float32)

        if feature.shape[0] < max_len:
            padding = torch.zeros((max_len - feature.shape[0], feature.shape[1]))
            feature = torch.cat((feature, padding), dim=0)
        else:
            feature = feature[:max_len]
        return feature

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # Initialize file paths at the beginning to avoid UnboundLocalError
        audio_file = os.path.join(self.audio_path, item['audio_feature_path'])
        visual_file = os.path.join(self.visual_path, item['video_feature_path'])

        try:
            # Check if files exist before loading
            if not os.path.exists(audio_file):
                print(f"⚠️  Audio file not found: {audio_file}")
                # Create zero features as fallback
                audio_features = torch.zeros(self.config.max_length, self._get_audio_feature_dim())
            else:
                audio_features = np.load(audio_file)
                audio_features = self.pad_or_truncate(audio_features, self.config.max_length)

            if not os.path.exists(visual_file):
                print(f"⚠️  Visual file not found: {visual_file}")
                # Create zero features as fallback
                visual_features = torch.zeros(self.config.max_length, self._get_visual_feature_dim())
            else:
                visual_features = np.load(visual_file)
                visual_features = self.pad_or_truncate(visual_features, self.config.max_length)

            # Get personalized features - extract ID from filename
            # Extract person ID from audio or video filename (e.g., "001_001.npy" -> "001")
            try:
                if 'id' in item:
                    person_id = str(item['id'])
                else:
                    # Extract from filename - audio_feature_path format: "XXX_YYY.npy"
                    filename = item.get('audio_feature_path', item.get('video_feature_path', '000_000.npy'))
                    person_id = filename.split('_')[0]  # Get the person ID part
            except (KeyError, IndexError, AttributeError):
                person_id = '000'  # Default fallback ID

            if person_id in self.personalized_features:
                personalized_features = torch.tensor(self.personalized_features[person_id], dtype=torch.float32)
            else:
                personalized_features = torch.zeros(self.config.text_embed_dim, dtype=torch.float32)

            # Get label based on task type
            label_key = self.task_mapping[self.task_type]
            if label_key in item:
                label = torch.tensor(item[label_key], dtype=torch.long)
            else:
                # Fallback for missing label types (e.g., quinary not available in young dataset)
                print(f"⚠️ {label_key} not available, using binary classification instead")
                label = torch.tensor(item['bin_category'], dtype=torch.long)

            return {
                'audio_features': audio_features,
                'visual_features': visual_features,
                'personalized_features': personalized_features,
                'label': label,
                'person_id': person_id
            }

        except Exception as e:
            print(f"❌  Error loading item {idx}: {str(e)}")
            print(f"   Audio file: {audio_file}")
            print(f"   Visual file: {visual_file}")

            # Extract person ID safely for error case
            try:
                if 'id' in item:
                    person_id = str(item['id'])
                else:
                    filename = item.get('audio_feature_path', item.get('video_feature_path', '000_000.npy'))
                    person_id = filename.split('_')[0]
            except:
                person_id = '000'

            # Return zero features instead of crashing
            return {
                'audio_features': torch.zeros(self.config.max_length, self._get_audio_feature_dim()),
                'visual_features': torch.zeros(self.config.max_length, self._get_visual_feature_dim()),
                'personalized_features': torch.zeros(self.config.text_embed_dim, dtype=torch.float32),
                'label': torch.tensor(0, dtype=torch.long),  # Default label
                'person_id': person_id
            }

    def _get_audio_feature_dim(self):
        """Get audio feature dimension based on feature type"""
        return self.config.feature_dims.get(self.audio_feature_type, 512)

    def _get_visual_feature_dim(self):
        """Get visual feature dimension based on feature type"""
        return self.config.feature_dims.get(self.visual_feature_type, 1000)
