#  -*- coding: utf-8 -*-
#  Copyright 2019 Pierre-Luc Delisle. All Rights Reserved.
#  #
#  Licensed under the MIT License;
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://opensource.org/licenses/MIT
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ==============================================================================

from argparse import ArgumentParser


class ArgsParserType(object):
    MODEL_TRAINING = "training"


class ArgsParserFactory(object):

    @staticmethod
    def create_parser(parser_type):
        parser = ArgumentParser(description='DeepNormalize Training')
        parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=False)
        parser.add_argument("--amp-opt-level", dest="amp_opt_level", type=str, default="O1",
                            help="O0 - FP32 training, O1 - Mixed Precision (recommended), O2 - Almost FP16 Mixed Precision, O3 - FP16 Training.")
        parser.add_argument("--num-workers", dest="num_workers", default=2, type=int,
                            help="Number of data loading workers for each dataloader object (default: 2).")
        parser.add_argument("--local_rank", dest="local_rank", default=0, type=int, help="The local_rank of the GPU.")

        if parser_type is ArgsParserType.MODEL_TRAINING:
            parser.add_argument("--config-file", dest="config_file", required=True)
        return parser
