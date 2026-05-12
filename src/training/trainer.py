"""
Trainer Module - Cortex ML V2.1
Handles model training, hyperparameter optimization, and transfer learning.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_manager import DBManager
from src.features.feature_store import FeatureStore
from src.models.joint_corners_model import JointCornersModel


def train_joint_model() -> None:
    """Treino do modelo multimercado científico (JointCornersModel)."""
    print("\n" + "=" * 60)
    print("🧬 TREINO MULTIMERCADO CIENTÍFICO (Joint Model)")
    print("=" * 60)
    print("Este modo treina o modelo de 4 targets [h1H, a1H, h2H, a2H]")
    print("com walk-forward temporal e calibração por família.\n")
    db = DBManager()
    try:
        df = db.get_historical_data()
        if df.empty:
            raise ValueError("Banco sem histórico para treino.")

        feature_store = FeatureStore(db)
        X, _, timestamps = feature_store.get_training_features(df)
        y_joint = JointCornersModel.build_targets(df, X.index)

        joint = JointCornersModel()
        metrics = joint.train(X, y_joint, timestamps=timestamps if timestamps is not None else None)
        joint.save()

        print("✅ JointCornersModel treinado e salvo com sucesso.")
        print(
            f"MAE h1={metrics.mae_h1:.3f} | a1={metrics.mae_a1:.3f} | "
            f"h2={metrics.mae_h2:.3f} | a2={metrics.mae_a2:.3f}"
        )
    finally:
        db.close()

def train_model(args=None) -> None:
    """
    Treina o modelo de Machine Learning utilizando o pipeline Professional V2.
    """
    # Mantido como ponto único de entrada para treinos operacionais.
    # Regra atual solicitada pelo usuário: somente Joint Model.
    try:
        train_joint_model()
    except Exception as e:
        print(f"❌ Erro fatal no treinamento: {e}")
        traceback.print_exc()
