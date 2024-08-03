import unittest
from Modelo.model import get_content_based_recommendations_nn_unique, get_user_based_recommendations_svd

class TestRecommendationModel(unittest.TestCase):
    
    def test_content_based_recommendations(self):
        recommendations = get_content_based_recommendations_nn_unique("Atonement")
        self.assertIsInstance(recommendations, list)
    
    def test_collaborative_based_recommendations(self):
        recommendations = get_user_based_recommendations_svd(631)
        self.assertIsInstance(recommendations, list)

if __name__ == '__main__':
    unittest.main()
