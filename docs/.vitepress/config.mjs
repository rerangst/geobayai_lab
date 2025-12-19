import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(
  defineConfig({
    title: 'Nghiên cứu xView Challenge',
    description: 'Tài liệu nghiên cứu toàn diện về bộ dữ liệu ảnh vệ tinh xView và các giải pháp đạt giải',
    base: '/sen_doc/',
    lang: 'vi-VN',

    head: [
      ['meta', { name: 'theme-color', content: '#3eaf7c' }],
      ['meta', { name: 'apple-mobile-web-app-capable', content: 'yes' }],
    ],

    themeConfig: {
      nav: [
        { text: 'Trang chủ', link: '/' },
        { text: 'xView1', link: '/xview-challenges/xview1/dataset-xview1-detection' },
        { text: 'xView2', link: '/xview-challenges/xview2/dataset-xview2-xbd-building-damage' },
        { text: 'xView3', link: '/xview-challenges/xview3/dataset-xview3-sar-maritime' },
      ],

      sidebar: {
        '/xview-challenges/': [
          {
            text: 'Tổng quan',
            items: [
              { text: 'Mục lục', link: '/xview-challenges/' },
            ]
          },
          {
            text: 'xView1 - Phát hiện đối tượng',
            collapsed: false,
            items: [
              { text: 'Đặc tả bộ dữ liệu', link: '/xview-challenges/xview1/dataset-xview1-detection' },
              { text: 'Giải nhất - Reduced Focal Loss', link: '/xview-challenges/xview1/winner-1st-place-reduced-focal-loss' },
              { text: 'Giải nhì - ĐH Adelaide', link: '/xview-challenges/xview1/winner-2nd-place-university-adelaide' },
              { text: 'Giải ba - ĐH South Florida', link: '/xview-challenges/xview1/winner-3rd-place-university-south-florida' },
              { text: 'Giải tư - Studio Mapp', link: '/xview-challenges/xview1/winner-4th-place-studio-mapp' },
              { text: 'Giải năm - CMU SEI', link: '/xview-challenges/xview1/winner-5th-place-cmu-sei' },
            ]
          },
          {
            text: 'xView2 - Đánh giá thiệt hại công trình',
            collapsed: false,
            items: [
              { text: 'Đặc tả bộ dữ liệu (xBD)', link: '/xview-challenges/xview2/dataset-xview2-xbd-building-damage' },
              { text: 'Giải nhất - Siamese UNet', link: '/xview-challenges/xview2/winner-1st-place-siamese-unet' },
              { text: 'Giải nhì - Selim Sefidov', link: '/xview-challenges/xview2/winner-2nd-place-selim-sefidov' },
              { text: 'Giải ba - Eugene Khvedchenya', link: '/xview-challenges/xview2/winner-3rd-place-eugene-khvedchenya' },
              { text: 'Giải tư - Z-Zheng', link: '/xview-challenges/xview2/winner-4th-place-z-zheng' },
              { text: 'Giải năm - Dual-HRNet', link: '/xview-challenges/xview2/winner-5th-place-dual-hrnet' },
            ]
          },
          {
            text: 'xView3 - Phát hiện tàu biển',
            collapsed: false,
            items: [
              { text: 'Đặc tả bộ dữ liệu (SAR)', link: '/xview-challenges/xview3/dataset-xview3-sar-maritime' },
              { text: 'Giải nhất - CircleNet', link: '/xview-challenges/xview3/winner-1st-place-circlenet-bloodaxe' },
              { text: 'Giải nhì - Selim Sefidov', link: '/xview-challenges/xview3/winner-2nd-place-selim-sefidov' },
              { text: 'Giải ba - Tumenn', link: '/xview-challenges/xview3/winner-3rd-place-tumenn' },
              { text: 'Giải tư - AI2 Skylight', link: '/xview-challenges/xview3/winner-4th-place-ai2-skylight' },
              { text: 'Giải năm - Kohei', link: '/xview-challenges/xview3/winner-5th-place-kohei' },
            ]
          },
        ],
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/tchatb/sen_doc' }
      ],

      search: {
        provider: 'local'
      },

      outline: {
        level: [2, 3],
        label: 'Mục lục trang'
      },

      footer: {
        message: 'Tài liệu nghiên cứu xView Challenge Series',
        copyright: 'Tạo năm 2024'
      },

      docFooter: {
        prev: 'Trang trước',
        next: 'Trang sau'
      },

      lastUpdated: {
        text: 'Cập nhật lần cuối'
      },

      returnToTopLabel: 'Về đầu trang',
      sidebarMenuLabel: 'Menu',
      darkModeSwitchLabel: 'Giao diện'
    },

    mermaid: {
      theme: 'default'
    },

    mermaidPlugin: {
      class: 'mermaid'
    },

    markdown: {
      lineNumbers: true
    }
  })
)
