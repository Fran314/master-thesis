#ifndef __CNN_COMMON_H__
#define __CNN_COMMON_H__

#include <fstream>

const int resnet18_convs = 20;
const int resnet18_relus = 17;
const int resnet18_skips = 8;
const int resnet18_fc_idx = resnet18_convs;

const int resnet50_convs = 53;
const int resnet50_relus = 49;
const int resnet50_skips = 16;
const int resnet50_fc_idx = resnet50_convs;

const int resnet101_convs = 104;
const int resnet101_relus = 100;
const int resnet101_skips = 33;
const int resnet101_fc_idx = resnet101_convs;

const int vgg11_convs = 8;
const int vgg11_relus = 10;
const int vgg11_maxs = 5;
const int vgg11_fcs = 3;

void read_param(char *filename, size_t &padding, size_t &stride) {
  std::fstream file(filename, std::ios_base::in);
  if (!file.is_open()) {
    std::perror(filename);
    exit(1);
  }

  file >> padding >> stride;
  file.close();
}

#endif  // __CNN_COMMON_H__