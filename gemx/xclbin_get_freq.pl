#!/usr/bin/perl -w

# Copyright 2019 Xilinx, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

my $xclbin = shift();
defined($xclbin) && -f $xclbin || die("Usage:\n  $0 <xclbinfile>\n");

# head for older xclbins, tail for newer xclbin2

open(P, "strings $xclbin | grep MHz|") ||
  die("ERROR: Failed to open $xclbin\n");

my $l;
my $freq = 0;
my $freq1 = 0;
my $type = "none";
while (defined($l = <P>)) {
  if ($l =~ m/<clock port="DATA_CLK" frequency="(.+)MHz"/) {
    if ($freq == 0) {
      $freq = $1;
    }
  } elsif ($l =~ m/<core name="OCL_REGION_0" target="(.+)" type="clc_region" clockFreq="(.+)MHz"/) {
    ($type, $freq1) = ($1, $2);
  }
}
close(P);
if (($type ne "bitstream") && ($type ne "none")) {
  print "$type-$freq1\n";
} else {
  print "$freq\n";
}


