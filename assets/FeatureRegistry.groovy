[
    angles: [
        variants: [
            [suffix: '_phi', params: [angle_type: 'phi']],
            [suffix: '_psi', params: [angle_type: 'psi']],
        ]
    ],
    k_closest_distance: [
        variants: [
            [suffix: '_5d',  params: [res_num: 5]],
            [suffix: '_10d', params: [res_num: 10]],
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
            [suffix: '_7.5A', params: [threshold_distance: 7.5]],
            [suffix: '_10A',  params: [threshold_distance: 10.0]],
        ]
    ],
    rasa_residue: [
        variants: [
            [suffix: '_complex', params: [complex: true]],
            [suffix: '',         params: [complex: false]],
        ]
    ],
    sasa_residue: [
        variants: [
            [suffix: '_complex', params: [complex: true]],
            [suffix: '',         params: [complex: false]],
        ]
    ],
    sasa_structure: [
        variants: [
            [suffix: '_norm', params: [normalize: true]],
            [suffix: '',      params: [normalize: false]],
        ]
    ],
    bsa_structure: [
        variants: [
            [suffix: '_norm', params: [normalize: true]],
            [suffix: '',      params: [normalize: false]],
        ]
    ],
]