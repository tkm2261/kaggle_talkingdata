-- dmt_Train
SELECT
click_id,
ip,
sum_attr, last_attr, cnt_ip,
app,
device,
os,
channel,
is_attributed,
hour,
avg_app,
avg_device,
avg_os,
avg_channel,
avg_hour,
avg_ipdayhour,
avg_ip,
sum_ip,

cnt_app,
cnt_device,
cnt_os,
cnt_channel,
cnt_hour,
TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by ip order by click_time), SECOND) as click_diff_1,
TIMESTAMP_DIFF(click_time, LAG(click_time, 2) OVER(partition by ip order by click_time), SECOND) as click_diff_2,
TIMESTAMP_DIFF(click_time, LAG(click_time, 3) OVER(partition by ip order by click_time), SECOND) as click_diff_3,
TIMESTAMP_DIFF(click_time, LAG(click_time, 4) OVER(partition by ip order by click_time), SECOND) as click_diff_4,
TIMESTAMP_DIFF(click_time, LAG(click_time, 5) OVER(partition by ip order by click_time), SECOND) as click_diff_5,
LAG(channel, 1) OVER(partition by ip order by click_time) channel_1,
LAG(channel, 2) OVER(partition by ip order by click_time) channel_2,
LAG(avg_channel, 1) OVER(partition by ip order by click_time) avg_channel_1,
LAG(avg_channel, 2) OVER(partition by ip order by click_time) avg_channel_2
FROM
`talking.train_test3`
WHERE
click_id is null AND

--click_time <= '2017-11-08 16:00:00'


-- dmt_valid
SELECT
click_id,
ip,
sum_attr, last_attr, cnt_ip,
app,
device,
os,
channel,
is_attributed,
hour,
avg_app,
avg_device,
avg_os,
avg_channel,
avg_hour,
avg_ipdayhour,
avg_ip,
sum_ip,

cnt_app,
cnt_device,
cnt_os,
cnt_channel,
cnt_hour,
TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by ip order by click_time), SECOND) as click_diff_1,
TIMESTAMP_DIFF(click_time, LAG(click_time, 2) OVER(partition by ip order by click_time), SECOND) as click_diff_2,
TIMESTAMP_DIFF(click_time, LAG(click_time, 3) OVER(partition by ip order by click_time), SECOND) as click_diff_3,

LAG(channel, 1) OVER(partition by ip order by click_time) channel_1,
LAG(channel, 2) OVER(partition by ip order by click_time) channel_2,
LAG(avg_channel, 1) OVER(partition by ip order by click_time) avg_channel_1,
LAG(avg_channel, 2) OVER(partition by ip order by click_time) avg_channel_2
FROM
`talking.train_test3`
WHERE
click_id is null AND
  click_time >= '2017-11-09 04:00:00' AND
  click_time <= '2017-11-09 15:00:00'


  -- dmt_Test
  SELECT
  click_id,
  ip,
  sum_attr, last_attr, cnt_ip,
  app,
  device,
  os,
  channel,
  is_attributed,
  hour,
  avg_app,
  avg_device,
  avg_os,
  avg_channel,
  avg_hour,
  avg_ipdayhour,
  avg_ip,
  sum_ip,

  cnt_app,
  cnt_device,
  cnt_os,
  cnt_channel,
  cnt_hour,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 1) OVER(partition by ip order by click_time), SECOND) as click_diff_1,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 2) OVER(partition by ip order by click_time), SECOND) as click_diff_2,
  TIMESTAMP_DIFF(click_time, LAG(click_time, 3) OVER(partition by ip order by click_time), SECOND) as click_diff_3,

  LAG(channel, 1) OVER(partition by ip order by click_time) channel_1,
  LAG(channel, 2) OVER(partition by ip order by click_time) channel_2,
  LAG(avg_channel, 1) OVER(partition by ip order by click_time) avg_channel_1,
  LAG(avg_channel, 2) OVER(partition by ip order by click_time) avg_channel_2
  FROM
  `talking.train_test3`
  WHERE
    click_id is not null
