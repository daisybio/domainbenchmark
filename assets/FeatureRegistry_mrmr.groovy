[
    k_closest_distance: [
        variants: [
            [suffix: '_5k',  params: [res_num: 5]],
        ]
    ],
    patch_mean_contacts: [
        variants: [
            [suffix: '_5A',   params: [threshold_distance: 5.0]],
            [suffix: '_7.5A', params: [threshold_distance: 7.5]],
            [suffix: '_10A',  params: [threshold_distance: 10.0]],
        ]
    ],
    patch_max_size: [
        variants: [
            [suffix: '_5A',   params: [threshold_distance: 5.0]],
        ]
    ],
    sasa_structure: [
        variants: [
            [suffix: '',      params: [normalize: false]],
        ]
    ],
]