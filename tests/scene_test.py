import unittest
from oncrit.oncrit import ObjectState, Scene


class TestObjectState(unittest.TestCase):

    def test_object_state_creation(self):
        obj = ObjectState(
            id="obj1",
            x=10.0,
            y=20.0,
            vx=5.0,
            vy=5.0,
            psi_rad=1.57,
            ax=2.0,
            ay=2.0,
            width=2.0,
            length=5.0,
            timestamp=1234567890,
            classification="vehicle"
        )
        self.assertEqual(obj.id, "obj1")
        self.assertEqual(obj.position.x, 10.0)
        self.assertEqual(obj.position.y, 20.0)
        self.assertEqual(obj.vx, 5.0)
        self.assertEqual(obj.vy, 5.0)
        self.assertEqual(obj.psi_rad, 1.57)
        self.assertEqual(obj.ax, 2.0)
        self.assertEqual(obj.ay, 2.0)
        self.assertEqual(obj.width, 2.0)
        self.assertEqual(obj.length, 5.0)
        self.assertEqual(obj.timestamp, 1234567890)
        self.assertEqual(obj.classification, "vehicle")

    def test_object_state_default_values(self):
        obj = ObjectState(
            id="obj2",
            x=5.0,
            y=15.0,
            vx=0.0,
            vy=0.0,
            psi_rad=0.0
        )
        self.assertEqual(obj.ax, 0.0)
        self.assertEqual(obj.ay, 0.0)
        self.assertEqual(obj.width, 0.5)
        self.assertEqual(obj.length, 0.5)
        self.assertEqual(obj.timestamp, 0)
        self.assertEqual(obj.classification, "unknown")


class TestScene(unittest.TestCase):

    def setUp(self):
        self.obj1 = ObjectState(
            id="obj1",
            x=10.0,
            y=20.0,
            vx=5.0,
            vy=5.0,
            psi_rad=1.57,
            ax=2.0,
            ay=2.0,
            width=2.0,
            length=5.0,
            timestamp=1234567890,
            classification="vehicle"
        )
        self.obj2 = ObjectState(
            id="obj2",
            x=15.0,
            y=25.0,
            vx=6.0,
            vy=6.0,
            psi_rad=1.8,
            ax=3.0,
            ay=3.0,
            width=3.0,
            length=6.0,
            timestamp=1234567890,
            classification="car"
        )
        self.obj3 = ObjectState(
            id="obj3",
            x=5.0,
            y=15.0,
            vx=0.0,
            vy=0.0,
            psi_rad=0.0,
            ax=2.0,
            ay=2.0,
            width=2.0,
            length=5.0,
            timestamp=1234567890,
            classification="vehicle"
        )

    def test_scene_creation(self):
        scene = Scene([self.obj1, self.obj2], timestamp=1234567890)
        self.assertEqual(scene.timestamp, 1234567890)
        self.assertIn("obj1", scene.objects)
        self.assertIn("obj2", scene.objects)

    def test_scene_default_timestamp(self):
        scene = Scene([self.obj1])
        self.assertEqual(scene.timestamp, 1234567890)

    def test_get_object_ids(self):
        scene = Scene([self.obj1, self.obj2, self.obj3])
        self.assertListEqual(scene.get_object_ids(), ["obj1", "obj2", "obj3"])
        scene2 = Scene([self.obj3, self.obj2, self.obj1])
        self.assertListEqual(scene2.get_object_ids(), ["obj1", "obj2", "obj3"])


if __name__ == '__main__':
    unittest.main()