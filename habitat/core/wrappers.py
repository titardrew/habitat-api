import os

from habitat.core.logging import logger
from habitat.core.vector_env import VectorEnv, VectorWrapper
from habitat.core.visualizer import Visualizer
from habitat.config import Config
from habitat.utils.visualizations.utils import images_to_video


class VectorVideoRecorder(VectorWrapper):

    def __init__(
        self,
        vector_env: VectorEnv,
        vis_config: Config
    ) -> None:

        super().__init__(vector_env)
        self.visualizer = None

        self.directory = os.path.abspath(vis_config.DIR_PATH)
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)

        self.file_prefix = "vecenv"
        self.file_infix = '{}'.format(os.getpid())
        self.step_id = 0
        self.video_length = vis_config.VIDEO_LENGTH
        self.vis_cfg = vis_config

        self.recording = False
        self.recorded_frames = 0
        self.path = None
        self.images = []

    def start_visualizer(self):
        self.close_visualizer()

        fname = "{}.video.{}.video{:06}".format(self.file_prefix,
                                                self.file_infix,
                                                self.step_id)
        self.vid_name = fname
        self.visualizer = Visualizer(venv=self.vector_env,
                                     vis_config=self.vis_cfg)

        # image = self.visualizer.get_image()
        self.images = []  # [image]
        self.recorded_frames = 1
        self.recording = True

    def _video_enabled(self):
        return (self.step_id % self.vis_cfg.START_EVERY == 0
                and self.step_id >= self.vis_cfg.START_EVERY)

    def wait_step(self):
        observations = self.vector_env.wait_step()

        self.step_id += 1
        if self.recording:
            image = self.visualizer.get_image(observations)
            self.images.append(image)
            self.recorded_frames += 1
            if self.recorded_frames > self.video_length:
                logger.info("Saving video to ", self.directory)
                images_to_video(self.images,
                                self.directory,
                                self.vid_name)
                self.close_visualizer()
        elif self._video_enabled():
            self.start_visualizer()

        return observations

    def close_visualizer(self):
        self.recording = False
        self.recorded_frames = 0

    def close(self):
        VectorWrapper.close(self)
        self.close_visualizer()

    def __del__(self):
        self.close()
