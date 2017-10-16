#!/usr/bin/perl -w

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


