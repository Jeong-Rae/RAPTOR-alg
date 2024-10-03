import * as tf from '@tensorflow/tfjs';
import { Tensor2D } from '@tensorflow/tfjs';
import { UMAP } from 'umap-js';

// UMAP 기반 차원 축소 함수
async function globalClusterEmbeddings(
    embeddings: Tensor2D, // 입력 임베딩을 위한 TensorFlow.js 텐서
    dim: number, // 목표 차원 수
    n_neighbors: number | null = null, // 고려할 이웃 수
    metric: string = 'cosine' // UMAP을 위한 거리 측정 기준
): Promise<Tensor2D> {
    // UMAP-js 처리를 위해 텐서를 2D 배열로 변환
    const embeddingsArray = await embeddings.array();

    // n_neighbors가 제공되지 않은 경우 sqrt(임베딩 수)를 기본값으로 설정
    if (n_neighbors === null) {
        n_neighbors = Math.floor(Math.sqrt(embeddingsArray.length - 1));
    }

    // UMAP 차원 축소 설정
    const umap = new UMAP({
        nNeighbors: n_neighbors,
        nComponents: dim,
        metric: metric,
    });

    // 임베딩을 적합하고 낮은 차원으로 변환
    const reducedEmbeddings = umap.fit(embeddingsArray);

    // TensorFlow.js 텐서로 다시 변환
    const reducedTensor = tf.tensor2d(reducedEmbeddings);

    return reducedTensor;
}
