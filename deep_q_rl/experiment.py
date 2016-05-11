"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague
Modified by Norman Tasfi for PyGame Learning Environment (PLE)

"""
import logging
import numpy as np
from PIL import Image
import cv2

CROP_OFFSET = 8

class PLEExperiment(object):
    def __init__(self, ple, agent, resized_width, resized_height,
                 resize_method, num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng):
        self.ple = ple
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.frame_skip = frame_skip
        self.death_ends_episode = death_ends_episode
        self.min_action_set = ple.getActionSet()
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.resize_method = resize_method
        self.width, self.height = ple.getScreenDims()

        print self.resized_width, self.resized_height
        print self.width, self.height

        self.buffer_length = 2
        self.buffer_count = 0
        self.screen_buffer = np.empty((self.buffer_length,
                                       self.height, self.width),
                                      dtype=np.uint8)

        self.terminal_lol = False # Most recent episode ended on a loss of life
        self.max_start_nullops = max_start_nullops
        self.rng = rng

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                self.agent.start_testing()
                self.ple.display_screen = True
                self.run_epoch(epoch, self.test_length, True)
                self.ple.display_screen = False
                self.agent.finish_testing(epoch)

    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        self.terminal_lol = False # Make sure each epoch starts with a reset.
        steps_left = num_steps
        while steps_left > 0:
            prefix = "testing" if testing else "training"
            logging.info(prefix + " epoch: " + str(epoch) + " steps_left: " +
                         str(steps_left))
            _, num_steps = self.run_episode(steps_left, testing)

            steps_left -= num_steps


    def _init_episode(self):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""

        if not self.terminal_lol or self.ple.game_over():
            self.ple.reset_game()

            if self.max_start_nullops > 0:
                random_actions = self.rng.randint(0, self.max_start_nullops+1)
                for _ in range(random_actions):
                    self._act(self.ple.NOOP) # Null action

        #fill buffer up
        self._act(self.ple.NOOP)
        self._act(self.ple.NOOP)



    def _act(self, action):
        """Perform the indicated action for a single frame, return the
        resulting reward and store the resulting screen image in the
        buffer

        """
        reward = self.ple.act(action)
        index = self.buffer_count % self.buffer_length
        
        self.screen_buffer[index, ...] = np.transpose( self.ple.getScreenGrayscale() )

        self.buffer_count += 1
        return reward

    def _step(self, action):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        reward = 0
        for _ in range(self.frame_skip):
            reward += self._act(action)

        return reward

    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        self._init_episode()

        start_lives = self.ple.lives()

        action = self.agent.start_episode(self.get_observation())
        num_steps = 0
        while True:
            reward = self._step(self.min_action_set[action])
            self.terminal_lol = (self.death_ends_episode and not testing and
                                 self.ple.lives() < start_lives)
            terminal = self.ple.game_over() or self.terminal_lol
            num_steps += 1

            if num_steps%1000 == 0:
                print "num_steps", num_steps

            if terminal or num_steps >= max_steps:
                self.agent.end_episode(reward, terminal)
                break

            action = self.agent.step(reward, self.get_observation())
        return terminal, num_steps


    def get_observation(self):
        """ Resize and merge the previous two screen images """

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
       
        return self.resize_image(max_image)

    def resize_image(self, image):
        """ Appropriately resize a single image """

        #if the image is the same size, which we can control with PLE
        if image.shape[0] == self.width and image.shape[1] == self.resized_height:
            return image
        elif self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(image,
                                  (self.resized_width, resize_height),
                                  interpolation=cv2.INTER_LINEAR)

            if resize_height != self.resized_height:
                # Crop the part we want
                crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
                cropped = resized[crop_y_cutoff:
                                  crop_y_cutoff + self.resized_height, :]

                return cropped
            else:
                return resized

        elif self.resize_method == 'scale':
            #resized = np.array( Image.fromarray( image ).resize( (self.resized_width, self.resized_height),  resample=Image.BILINEAR) )
            resized = cv2.resize(image, (self.resized_width, self.resized_height), interpolation=cv2.INTER_LINEAR)
            return resized
        else:
            raise ValueError('Unrecognized image resize method.')

