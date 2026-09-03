import os
import sys
import time
import json

# Flat-import style used throughout this repo
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _sub in ("BaseLayers", "CombinedLayers", "Encoder", "fovi"):
    _path = os.path.join(_THIS_DIR, _sub)
    if _path not in sys.path:
        sys.path.append(_path)

try:
    import cupy as cp
except ImportError:
    import numpy as cp
import numpy as np

from vit_input_embedding import ViTInputEmbedding
from vit_encoder import VisionTransformerEncoder
from layer_normalization import LayerNormalization
from linear import Linear
from softmax import Softmax
from loss import CrossEntropy

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class VisionTransformer():
    """
    Image classification Vision Transformer (ViT), built from from-scratch primitives:
        images -> PatchEmbedding -> [CLS + patches] + pos_embedding
               -> N x EncoderBlock (self-attention, no mask)
               -> LayerNorm
               -> take CLS token representation
               -> Linear classification head -> Softmax -> class probabilities
    """

    def __init__(self, optimizer, img_size=224, patch_size=16, in_channels=3, num_classes=100,
                 d_model=64, d_ff=128, num_attention_heads=4, block_num=2, 
                 tokenizer_type='fovi', dropout_rate=0.1, label_smoothing=0.1, data_type=cp.float32):
        self.optimizer = optimizer
        self.data_type = data_type
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_attention_heads = num_attention_heads
        self.block_num = block_num
        self.tokenizer_type = tokenizer_type
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing

        self.embedding = ViTInputEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=in_channels, 
            d_model=d_model,
            optimizer=optimizer, 
            tokenizer_type=tokenizer_type,
            dropout_rate=dropout_rate, 
            data_type=data_type
        )
        self.encoder = VisionTransformerEncoder(
            optimizer=optimizer, 
            d_model=d_model, 
            d_ff=d_ff, 
            num_attention_heads=num_attention_heads,
            block_num=block_num, 
            dropout_rate=dropout_rate, 
            data_type=data_type
        )
        self.layernorm = LayerNormalization(
            optimizer=optimizer, 
            normalized_shape=d_model, 
            eps=1e-5, 
            data_type=data_type
        )
        self.classifier = Linear(
            in_features=d_model, 
            out_features=num_classes, 
            optimizer=optimizer, 
            use_bias=True, 
            data_type=data_type
        )
        self.softmax = Softmax()
        self.cross_entropy = CrossEntropy(padding_id=-1, vocab_size=num_classes, label_smoothing=label_smoothing)

        self.num_patches = self.embedding.num_patches
        self.seq_len = self.embedding.seq_len
        self.class_to_idx = None

    def forward(self, images, training=True):
        # images: (B, C, H, W)
        x = self.embedding.forward(images, training)   # (B, seq_len, d_model)
        x = self.encoder.forward(x, training)           # (B, seq_len, d_model)
        x = self.layernorm.forward(x)
        self.encoder_output_shape = x.shape

        cls_repr = x[:, 0, :]                           # (B, d_model)
        logits = self.classifier.forward(cls_repr)      # (B, num_classes)

        self.probs = self.softmax.forward(logits)
        return self.probs

    def compute_loss(self, labels):
        # labels: (B,) integer class ids
        loss = self.cross_entropy.forward(self.probs, labels)
        return loss.mean()

    def backward(self, labels):
        grad_logits = self.cross_entropy.grad(self.probs, labels)     # (B, num_classes)
        grad_cls = self.classifier.backward(grad_logits)               # (B, d_model)

        grad_x = cp.zeros(self.encoder_output_shape, dtype=grad_cls.dtype)
        grad_x[:, 0, :] = grad_cls                                     # only CLS token receives loss grad

        grad_x = self.layernorm.backward(grad_x)
        grad_x = self.encoder.backward(grad_x)
        grad_images = self.embedding.backward(grad_x)
        return grad_images

    def update_weights(self):
        self.classifier.update_weights()
        self.layernorm.update_weights()
        self.encoder.update_weights()
        self.embedding.update_weights()

    def predict(self, images):
        probs = self.forward(images, training=False)
        return cp.argmax(probs, axis=-1)

    def calculate_trainable_parameters(self):
        """
        Calculates total and trainable parameters in the model.
        """
        total_params = 0
        
        # 1. Embedding layer
        total_params += self.embedding.cls_token.size
        total_params += self.embedding.pos_embedding.size
        if self.tokenizer_type == 'standard':
            total_params += self.embedding.patch_embed.weights.size
            total_params += self.embedding.patch_embed.bias.size
        elif self.tokenizer_type == 'fovi':
            try:
                import torch
                total_params += sum(p.numel() for p in self.embedding.patch_embed.parameters() if p.requires_grad)
            except Exception:
                pass
                
        # 2. Encoder layers
        for layer in self.encoder.encoder_layers:
            # LayerNorm 1 & 2
            total_params += layer.layernorm1.gamma.size + layer.layernorm1.beta.size
            total_params += layer.layernorm2.gamma.size + layer.layernorm2.beta.size
            # MultiHeadAttention (W_q, W_k, W_v, W_o)
            total_params += layer.multi_head_attention.W_q.weights.size
            total_params += layer.multi_head_attention.W_k.weights.size
            total_params += layer.multi_head_attention.W_v.weights.size
            total_params += layer.multi_head_attention.W_o.weights.size + layer.multi_head_attention.W_o.bias.size
            # FFN (fc1, fc2)
            total_params += layer.ffn.fc1.weights.size + layer.ffn.fc1.bias.size
            total_params += layer.ffn.fc2.weights.size + layer.ffn.fc2.bias.size
            
        # 3. Final LayerNorm & Classifier
        total_params += self.layernorm.gamma.size + self.layernorm.beta.size
        total_params += self.classifier.weights.size + self.classifier.bias.size
        
        return int(total_params)

    def calculate_macs_and_flops(self, input_shape=None):
        """
        Calculates theoretical Multiply-Accumulate Operations (MACs) and FLOPs
        for a single forward pass with the given input shape.
        """
        C = self.in_channels
        P = self.patch_size
        D = self.d_model
        D_ff = self.d_ff
        N_patches = self.num_patches
        N_total = self.seq_len  # N_patches + 1
        L = self.block_num
        K = self.num_classes

        # 1. Patch Embedding MACs
        patch_macs = N_patches * (C * P * P * D)
        
        # 2. Per-Block Attention MACs:
        # Q, K, V projections: 3 * N_total * D * D
        # Attention scores (Q @ K^T): N_total * N_total * D
        # Attention context (Scores @ V): N_total * N_total * D
        # Output projection: N_total * D * D
        mha_macs = (4 * N_total * D * D) + (2 * N_total * N_total * D)
        
        # 3. Per-Block FFN MACs:
        # FC1 (D -> D_ff): N_total * D * D_ff
        # FC2 (D_ff -> D): N_total * D_ff * D
        ffn_macs = 2 * N_total * D * D_ff
        
        # Total for L blocks
        blocks_macs = L * (mha_macs + ffn_macs)
        
        # 4. Classifier Head MACs (CLS token -> Classes)
        classifier_macs = D * K
        
        total_macs = int(patch_macs + blocks_macs + classifier_macs)
        total_flops = int(2 * total_macs)
        
        return total_macs, total_flops

    def evaluate(self, dataloader, desc="Evaluating"):
        """
        Evaluates the model over the given dataloader.
        Returns:
            dict containing accuracy, loss, total_time, throughput, num_samples
        """
        total_correct = 0
        total_loss = 0.0
        total_samples = 0
        num_batches = 0
        
        start_time = time.time()
        total_batches = len(dataloader) if hasattr(dataloader, '__len__') else None
        pbar = tqdm(dataloader, total=total_batches, desc=desc, file=sys.stdout, dynamic_ncols=True, leave=False)
        for batch_images, batch_labels in pbar:
            batch_images = cp.array(batch_images)
            batch_labels = cp.array(batch_labels)
            
            probs = self.forward(batch_images, training=False)
            loss = self.compute_loss(batch_labels)
            if hasattr(loss, "get"):
                loss = loss.get()
            total_loss += float(loss)
            
            preds = cp.argmax(probs, axis=-1)
            correct = int(cp.sum(preds == batch_labels).get() if hasattr(preds, "get") else cp.sum(preds == batch_labels))
            total_correct += correct
            total_samples += len(batch_labels)
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{total_loss / max(1, num_batches):.4f}',
                'acc': f'{(total_correct / max(1, total_samples)) * 100.0:.2f}%'
            })

        total_time = time.time() - start_time
        accuracy = (total_correct / max(1, total_samples)) * 100.0
        avg_loss = total_loss / max(1, num_batches)
        throughput = total_samples / max(1e-6, total_time)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'total_time': total_time,
            'throughput': throughput,
            'total_samples': total_samples
        }

    def save_checkpoint(self, filepath):
        """
        Saves all model weights, optimizer state, class mappings, and config to a .npz file.
        """
        state = {}
        
        # Save architecture config
        config = {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'in_channels': self.in_channels,
            'num_classes': self.num_classes,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_attention_heads': self.num_attention_heads,
            'block_num': self.block_num,
            'tokenizer_type': self.tokenizer_type,
            'dropout_rate': self.dropout_rate,
            'class_to_idx': self.class_to_idx
        }
        state['config_json'] = np.array([json.dumps(config)])

        # Save embeddings
        state['cls_token'] = cp.asnumpy(self.embedding.cls_token) if hasattr(cp, 'asnumpy') else self.embedding.cls_token
        state['pos_embedding'] = cp.asnumpy(self.embedding.pos_embedding) if hasattr(cp, 'asnumpy') else self.embedding.pos_embedding
        
        if self.tokenizer_type == 'standard':
            state['patch_w'] = cp.asnumpy(self.embedding.patch_embed.weights) if hasattr(cp, 'asnumpy') else self.embedding.patch_embed.weights
            state['patch_b'] = cp.asnumpy(self.embedding.patch_embed.bias) if hasattr(cp, 'asnumpy') else self.embedding.patch_embed.bias
        elif self.tokenizer_type == 'fovi':
            try:
                for name, param in self.embedding.patch_embed.named_parameters():
                    state[f'fovi_{name}'] = param.detach().cpu().numpy()
            except Exception as e:
                print(f"[WARNING] Could not save FOVI patch embed parameters: {e}")

        # Save Encoder blocks
        for i, block in enumerate(self.encoder.encoder_layers):
            state[f'b{i}_ln1_gamma'] = cp.asnumpy(block.layernorm1.gamma) if hasattr(cp, 'asnumpy') else block.layernorm1.gamma
            state[f'b{i}_ln1_beta'] = cp.asnumpy(block.layernorm1.beta) if hasattr(cp, 'asnumpy') else block.layernorm1.beta
            state[f'b{i}_ln2_gamma'] = cp.asnumpy(block.layernorm2.gamma) if hasattr(cp, 'asnumpy') else block.layernorm2.gamma
            state[f'b{i}_ln2_beta'] = cp.asnumpy(block.layernorm2.beta) if hasattr(cp, 'asnumpy') else block.layernorm2.beta

            state[f'b{i}_wq'] = cp.asnumpy(block.multi_head_attention.W_q.weights) if hasattr(cp, 'asnumpy') else block.multi_head_attention.W_q.weights
            state[f'b{i}_wk'] = cp.asnumpy(block.multi_head_attention.W_k.weights) if hasattr(cp, 'asnumpy') else block.multi_head_attention.W_k.weights
            state[f'b{i}_wv'] = cp.asnumpy(block.multi_head_attention.W_v.weights) if hasattr(cp, 'asnumpy') else block.multi_head_attention.W_v.weights
            state[f'b{i}_wo'] = cp.asnumpy(block.multi_head_attention.W_o.weights) if hasattr(cp, 'asnumpy') else block.multi_head_attention.W_o.weights
            state[f'b{i}_wo_b'] = cp.asnumpy(block.multi_head_attention.W_o.bias) if hasattr(cp, 'asnumpy') else block.multi_head_attention.W_o.bias

            state[f'b{i}_fc1_w'] = cp.asnumpy(block.ffn.fc1.weights) if hasattr(cp, 'asnumpy') else block.ffn.fc1.weights
            state[f'b{i}_fc1_b'] = cp.asnumpy(block.ffn.fc1.bias) if hasattr(cp, 'asnumpy') else block.ffn.fc1.bias
            state[f'b{i}_fc2_w'] = cp.asnumpy(block.ffn.fc2.weights) if hasattr(cp, 'asnumpy') else block.ffn.fc2.weights
            state[f'b{i}_fc2_b'] = cp.asnumpy(block.ffn.fc2.bias) if hasattr(cp, 'asnumpy') else block.ffn.fc2.bias

        # Save Final LayerNorm & Classifier
        state['ln_final_gamma'] = cp.asnumpy(self.layernorm.gamma) if hasattr(cp, 'asnumpy') else self.layernorm.gamma
        state['ln_final_beta'] = cp.asnumpy(self.layernorm.beta) if hasattr(cp, 'asnumpy') else self.layernorm.beta
        state['classifier_w'] = cp.asnumpy(self.classifier.weights) if hasattr(cp, 'asnumpy') else self.classifier.weights
        state['classifier_b'] = cp.asnumpy(self.classifier.bias) if hasattr(cp, 'asnumpy') else self.classifier.bias

        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or '.', exist_ok=True)
        np.savez(filepath, **state)
        print(f"Checkpoint successfully saved to: {filepath}")

    def load_checkpoint(self, filepath):
        """
        Loads weights from a .npz checkpoint file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
            
        data = np.load(filepath, allow_pickle=True)
        
        if 'config_json' in data:
            try:
                config = json.loads(str(data['config_json'][0]))
                if 'class_to_idx' in config and config['class_to_idx'] is not None:
                    self.class_to_idx = config['class_to_idx']
            except Exception:
                pass

        if 'cls_token' in data:
            self.embedding.cls_token = cp.array(data['cls_token'], dtype=self.data_type)
        if 'pos_embedding' in data:
            self.embedding.pos_embedding = cp.array(data['pos_embedding'], dtype=self.data_type)
        if self.tokenizer_type == 'standard' and 'patch_w' in data:
            self.embedding.patch_embed.weights = cp.array(data['patch_w'], dtype=self.data_type)
            self.embedding.patch_embed.bias = cp.array(data['patch_b'], dtype=self.data_type)
        elif self.tokenizer_type == 'fovi':
            try:
                import torch
                fovi_state = {}
                for name, param in self.embedding.patch_embed.named_parameters():
                    key = f'fovi_{name}'
                    if key in data:
                        fovi_state[name] = torch.tensor(data[key], device=self.embedding.device, dtype=param.dtype)
                if fovi_state:
                    self.embedding.patch_embed.load_state_dict(fovi_state, strict=False)
            except Exception as e:
                print(f"[WARNING] Could not load FOVI patch embed parameters: {e}")

        for i, block in enumerate(self.encoder.encoder_layers):
            if f'b{i}_ln1_gamma' in data:
                block.layernorm1.gamma = cp.array(data[f'b{i}_ln1_gamma'], dtype=self.data_type)
                block.layernorm1.beta = cp.array(data[f'b{i}_ln1_beta'], dtype=self.data_type)
            if f'b{i}_ln2_gamma' in data:
                block.layernorm2.gamma = cp.array(data[f'b{i}_ln2_gamma'], dtype=self.data_type)
                block.layernorm2.beta = cp.array(data[f'b{i}_ln2_beta'], dtype=self.data_type)

            if f'b{i}_wq' in data:
                block.multi_head_attention.W_q.weights = cp.array(data[f'b{i}_wq'], dtype=self.data_type)
                block.multi_head_attention.W_k.weights = cp.array(data[f'b{i}_wk'], dtype=self.data_type)
                block.multi_head_attention.W_v.weights = cp.array(data[f'b{i}_wv'], dtype=self.data_type)
                block.multi_head_attention.W_o.weights = cp.array(data[f'b{i}_wo'], dtype=self.data_type)
                block.multi_head_attention.W_o.bias = cp.array(data[f'b{i}_wo_b'], dtype=self.data_type)

            if f'b{i}_fc1_w' in data:
                block.ffn.fc1.weights = cp.array(data[f'b{i}_fc1_w'], dtype=self.data_type)
                block.ffn.fc1.bias = cp.array(data[f'b{i}_fc1_b'], dtype=self.data_type)
                block.ffn.fc2.weights = cp.array(data[f'b{i}_fc2_w'], dtype=self.data_type)
                block.ffn.fc2.bias = cp.array(data[f'b{i}_fc2_b'], dtype=self.data_type)

        if 'ln_final_gamma' in data:
            self.layernorm.gamma = cp.array(data['ln_final_gamma'], dtype=self.data_type)
            self.layernorm.beta = cp.array(data['ln_final_beta'], dtype=self.data_type)
        if 'classifier_w' in data:
            self.classifier.weights = cp.array(data['classifier_w'], dtype=self.data_type)
            self.classifier.bias = cp.array(data['classifier_b'], dtype=self.data_type)
            
        print(f"Loaded checkpoint weights from: {filepath}")

    def fit_generator(self, num_epochs, dataloader, val_dataloader=None, checkpoint_path=None):
        """
        Train using a data generator, optionally evaluating on validation set
        and saving the best checkpoint.
        """
        best_val_acc = -1.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        if hasattr(dataloader, 'class_to_idx'):
            self.class_to_idx = dataloader.class_to_idx

        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            epoch_correct = 0
            num_samples = 0
            num_batches = 0
            epoch_start = time.time()

            total_batches = len(dataloader) if hasattr(dataloader, '__len__') else None
            pbar = tqdm(dataloader, total=total_batches, desc=f"Epoch {epoch:02d}/{num_epochs:02d} [Train]", file=sys.stdout, dynamic_ncols=True, leave=True)
            for batch_images, batch_labels in pbar:
                batch_images = cp.array(batch_images)
                batch_labels = cp.array(batch_labels)

                probs = self.forward(batch_images, training=True)
                loss = self.compute_loss(batch_labels)
                if hasattr(loss, "get"):
                    loss = loss.get()
                epoch_loss += float(loss)

                preds = cp.argmax(probs, axis=-1)
                correct = int(cp.sum(preds == batch_labels).get() if hasattr(preds, "get") else cp.sum(preds == batch_labels))
                epoch_correct += correct
                num_samples += len(batch_labels)
                num_batches += 1

                self.backward(batch_labels)
                self.update_weights()
                self.optimizer._step()

                pbar.set_postfix({
                    'loss': f'{epoch_loss / max(1, num_batches):.4f}',
                    'acc': f'{(epoch_correct / max(1, num_samples)) * 100.0:.2f}%',
                    'lr': f'{self.optimizer.lr:.6f}'
                })

            avg_loss = epoch_loss / max(1, num_batches)
            accuracy = (epoch_correct / max(1, num_samples)) * 100.0
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)

            elapsed = time.time() - epoch_start
            log_str = f"Epoch {epoch:02d}/{num_epochs:02d} Summary - Time: {elapsed:.2f}s - Train Loss: {avg_loss:.4f} - Train Acc: {accuracy:.2f}%"

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader, desc=f"Epoch {epoch:02d}/{num_epochs:02d} [Val]")
                history['val_loss'].append(val_metrics['loss'])
                history['val_acc'].append(val_metrics['accuracy'])
                log_str += f" | Val Loss: {val_metrics['loss']:.4f} - Val Acc: {val_metrics['accuracy']:.2f}%"

                if checkpoint_path and val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint(checkpoint_path)
                    # Automatically keep best_model.npz synced with the highest val accuracy
                    best_model_fallback = os.path.join(os.path.dirname(checkpoint_path) or '.', 'best_model.npz')
                    if os.path.abspath(checkpoint_path) != os.path.abspath(best_model_fallback):
                        try:
                            import shutil
                            shutil.copyfile(checkpoint_path, best_model_fallback)
                        except Exception:
                            pass
            elif checkpoint_path:
                self.save_checkpoint(checkpoint_path)

            print(log_str)

        return history

