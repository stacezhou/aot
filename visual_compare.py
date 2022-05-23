from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.layouts import layout
from bokeh.models import ColumnDataSource,ImageRGBA,Slider,Button,AutocompleteInput
from bokeh.models import CheckboxButtonGroup
from bokeh.models import TextAreaInput

from pathlib import Path
from PIL import Image
import numpy as np

# 辅助函数
def read_rgba_image(image_file):
    return Image.open(image_file).convert('RGBA').__array__()

def read_rgba_mask(mask_file):
    mask = read_rgba_image(mask_file)
    mask[mask[...,:3].sum(axis=2) == 0] = 0
    return mask

def rgba(im):
    return np.flip(np.require(im,np.uint8,['C']).view(dtype=np.uint32).squeeze(),axis=0)

def update_videos(data):
    data['masks_label'] = ['pred','ref']
    data['videos'] = [v.name for v in data['pred_root'].iterdir() if v.is_dir()]
    print(len(data['videos']))
    data['frames'] = dict()
    for v in data['videos']:
        frames_name = [f.name for f in (data['gt_root'] / v).glob('*.png')]
        frames = [f.stem for f in (data['pred_root'] / v).glob('*.png') if f.name in frames_name]
        data['frames'][v] = frames
    

image_root = Path('/data/YouTube/test/JPEGImages')
data = dict()
data['pred_root'] = Path(
    'results/baseline851_SwinB_AOTL__PRE_YTB_DAV/eval/youtubevos2019_test_baseline851_SwinB_AOTL_PRE_YTB_DAV_ckpt_unknown_ema_flip_ms_0dot75_1dot0_1dot4'
    ) / 'Annotations'
data['gt_root'] = Path(
    './results/sub_851/sub_851'
    ) / 'Annotations'
update_videos(data)
import pandas as pd
frame_df_file = \
    '/home/zh21/code/aot/results/baseline851_SwinB_AOTL__PRE_YTB_DAV/eval/frame_df_youtubevos2019_test_baseline851_SwinB_AOTL_PRE_YTB_DAV_ckpt_unknown_ema_flip_ms_0dot75_1dot0_1dot4_vs_sub_851.csv'

frame_df = pd.read_csv(
    frame_df_file
).reset_index(drop=True).set_index('name')
video_df_file = str(Path(frame_df_file).parent / ('video_' + Path(frame_df_file).name[6:]))
print(pd.read_csv(video_df_file).head(10))
# 模块
videoSelectInput = AutocompleteInput(completions=data['videos'], value=data['videos'][0])
frameSlider = Slider(value=0)
maskSelect = CheckboxButtonGroup(labels=data['masks_label'],active=[0,1])
W = 600
H = 400

plot = figure(width=W,height=H)
plot.x_range.update(
    start=0,
    end=W
)
plot.y_range.update(
    start=0,
    end=H
)
imageGlyph = ImageRGBA(x=0,y=0,dw=W,dh=H)
maskGlyph = ImageRGBA(x=0,y=0,dw=W,dh=H)
gtmaskGlyph = ImageRGBA(x=0,y=0,dw=W,dh=H)

imageSource = ColumnDataSource()
maskSource = ColumnDataSource()
gtmaskSource = ColumnDataSource()
plot.add_glyph(imageSource,imageGlyph)
plot.add_glyph(maskSource,maskGlyph)
plot.add_glyph(gtmaskSource,gtmaskGlyph)

lastFrame = Button(label='上一帧')
nextFrame = Button(label='下一帧')
lastVideo = Button(label='上个视频')
nextVideo = Button(label='下个视频')
refresh = Button(label='刷新')
compare = Button(label='切换mask显示')
img_name = Button(label='')
text_area_input = TextAreaInput(value="", title="视频子集",max_length=100000, rows = 3)


# 动作函数
def change_frame():
    i = frameSlider.value
    frameSlider.update(
        start = 0,
        end = len(data['frames'][videoSelectInput.value]) - 1
    )
    
    v = videoSelectInput.value
    image = read_rgba_image(str(image_root / v / data['frames'][v][i]) + '.jpg')
    mask = read_rgba_mask(str(data['pred_root'] / v /data['frames'][v][i]) + '.png')
    imageSource.data['image'] = [rgba(image)]
    img_name.label = f"{data['frames'][v][i]}"
    if 0 in maskSelect.active:
        maskGlyph.global_alpha = 0.70
    else:
        maskGlyph.global_alpha = 0
    maskSource.data['image'] = [rgba(mask)]

    gtmask_path = data['gt_root'] / v / (data['frames'][v][i] + '.png')
    if 1 in maskSelect.active and gtmask_path.exists():
        gtmaskGlyph.global_alpha = 0.70
        gtmask = read_rgba_mask(str(data['gt_root'] / v /data['frames'][v][i]) + '.png')
        gtmaskSource.data['image'] = [rgba(gtmask)]
    else:
        gtmaskGlyph.global_alpha = 0


def subset():
    if text_area_input.value == '':
        videoSelectInput.completions = data['videos']
        videoSelectInput.value = data['videos'][0]
        change_frame()
        return

    videos = text_area_input.value.split()
    videos = [v for vs in videos for v in vs.split(',')]
    videos = [v.strip(',') for v in videos]
    videos = [v.strip('\'') for v in videos]
    videos = [v.strip('\"') for v in videos]
    videos = [v.strip('{"') for v in videos]
    videos = [v.strip('}"') for v in videos]
    videos = [v.strip(']"') for v in videos]
    videos = [v.strip('["') for v in videos]
    print(videos)
    videos = [v for v in videos if v in data['videos']]
    videoSelectInput.completions = videos
    videoSelectInput.value = videos[0]
    text_area_input.value = str(videos)
    change_frame()

def next_video():
    frameSlider.value = 0
    i = videoSelectInput.completions.index(videoSelectInput.value)
    if i != len(videoSelectInput.completions) - 1:
        videoSelectInput.value = videoSelectInput.completions[i+1]
    print(frame_df.loc[videoSelectInput.value].values)
    
    change_frame()

def last_video():
    frameSlider.value = 0
    i = videoSelectInput.completions.index(videoSelectInput.value)
    if i != 0:
        videoSelectInput.value = videoSelectInput.completions[i-1]
    
    print(frame_df.loc[videoSelectInput.value].values)
    change_frame()

    
def last_frame():
    if frameSlider.value == 0:
        return 
    else:
        frameSlider.value -= 1
    
    change_frame()

def next_frame():
    if frameSlider.value == frameSlider.end:
        return
    else:
        frameSlider.value += 1
    
    change_frame()

# 动作响应绑定
def do_compare():
    active = maskSelect.active
    new_active = []
    if 0 not in active:
        new_active.append(0)
    if 1 not in active:
        new_active.append(1)
    maskSelect.active = new_active
    change_frame()

frameSlider.on_change( 'value', lambda a,o,n: change_frame())
videoSelectInput.on_change('value',lambda a,o,n:change_frame())
maskSelect.on_change('active', lambda a,o,n:change_frame())
refresh.on_click(lambda a:subset())
lastFrame.on_click(lambda a:last_frame())
nextFrame.on_click(lambda a:next_frame())
lastVideo.on_click(lambda a:last_video())
nextVideo.on_click(lambda a:next_video())
compare.on_click(lambda a:do_compare())

# 状态初始化
frameSlider.update(
    value = 0,
    start = 0,
    end = 1
)
change_frame()


# 布局
frameSlider.width = 500
curdoc().add_root(layout([
     [plot,[
        img_name,
        videoSelectInput, 
        text_area_input,
        lastVideo,
        nextVideo,
        refresh
     ]],
    [maskSelect, compare],
    [lastFrame,nextFrame], 
    frameSlider,
]))
