import datalib.transforms as DT
import torchvision.transforms as T
from datalib.dataset import JonathanMarkerDataset


image_transform = bbox_transform = transform = DT.Compose([
    DT.exif_transpose(),
    DT.squish_resize(224),
    DT.to_tensor(),
    DT.normalize(mode="imagenet"),
])

class DFApply:
    def encode_label(
        label_name="label",
        class_map={
            "1도": 0,
            "표재성 2도": 0,
            "심재성 2도": 1,
            "3도": 1,
            "미상": None,
            "불일치": None,
        },
    ):
        def _f(df):
            df[label_name] = df["label_id"].replace(class_map)
            return df
        return _f

    def drop_na():
        def _f(df):
            df = df.dropna()
            return df
        return _f
    
    def coord_to_int(columns=["x1","x2","y1","y2"]):
        def _f(df):
            df.loc[:,columns] = df.loc[:,columns].astype(int)
            return df
        return _f

    def set_constant(
            column_name="",
            value="",
        ):
        def _f(df):
            df[column_name] = value
            return df
        return _f


class ReturnTransform:
    def get_image(transform=None):
        def _f(item):
            image, row = item
            if transform is not None:
                image = transform(image)
            return image
        return _f

    def get_bbox(transform=None):
        def _f(item):
            image, row = item
            coords = (row["x1"], row["y1"], row["x2"], row["y2"]) 
            bbox = image.crop(coords)
            if transform is not None:
                bbox = transform(bbox)
            return bbox
        return _f

    def get_meta():
        def _f(item):
            image, row = item
            return row
        return _f

    def get_label():
        def _f(item):
            image, row = item
            return row["label"]
        return _f



df_pipeline = DT.Compose([
    DFApply.encode_label(),
    DFApply.drop_na(),
    DFApply.coord_to_int(),
])

dataset = JonathanMarkerDataset(
    name="image_2000_B",
    df_apply=df_pipeline,
    image_column="file_name",
    image_convert="RGB",
    return_transform=[
        ReturnTransform.get_image(transform=transform),
        ReturnTransform.get_bbox(transform=transform),
        ReturnTransform.get_label(),
        ReturnTransform.get_meta(),
    ]
)


if __name__ == '__main__':
    print(dataset.df)
    print(dataset[0])